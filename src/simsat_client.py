"""DPhi SimSat API client — fetches Sentinel-2 tiles from the platform.

This is the integration surface specified by the hackathon's "Use of
Satellite Imagery" judging criterion. SimSat is launched locally via
`docker compose up` from https://github.com/DPhi-Space/SimSat and exposes:

    GET  http://localhost:9005/data/current/position
    GET  http://localhost:9005/data/current/image/sentinel
    GET  http://localhost:9005/data/image/sentinel
                ?lon=…&lat=…&timestamp=…&spectral_bands=…&size_km=…&return_type=png

Pathogen Scout's three-tier agent operates on (B4, B8, B8A) — i.e.
(red, nir, rededge3) in SimSat's parlance. We fetch each band as a
separate PNG and stack them into the .npz tile format that the rest of
the pipeline already consumes. This keeps `agent_logic`, `multispectral`,
and `classifier` unchanged — the adapter is the *only* code path that
talks to the platform.

Usage:
    # Fetch a tile at the satellite's current position:
    python -m src.simsat_client --current --out data/sample_tiles/live.npz

    # Fetch a specific lat/lon/time:
    python -m src.simsat_client --lat 46.5197 --lon 6.6323 \
        --timestamp 2026-03-01T16:00:00Z --out data/sample_tiles/lausanne.npz

    # Pipe straight into the agent:
    python -m src.simsat_client --current --out /tmp/t.npz && \
        python -m src.agent_logic --tile /tmp/t.npz --verbose
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import requests

SIMSAT_BASE_URL = "http://localhost:9005"
REQUEST_TIMEOUT_SECONDS = 30

# Pathogen Scout NDRE math expects (B4 red, B8 nir, B8A rededge).
# SimSat's spectral_bands vocabulary names them (red, nir, rededge3).
PATHOGEN_SCOUT_BANDS: dict[str, str] = {
    "B4": "red",
    "B8": "nir",
    "B8A": "rededge3",
}


@dataclass
class SimSatTile:
    """A multispectral tile fetched from SimSat, ready for agent_logic."""

    bands: dict[str, np.ndarray]   # keys: B4, B8, B8A — float32 in [0, 1]
    lat: float
    lon: float
    timestamp: str
    size_km: float
    metadata: dict                 # raw sentinel_metadata for the last band

    def save_npz(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            B4=self.bands["B4"],
            B8=self.bands["B8"],
            B8A=self.bands["B8A"],
            _lat=np.array([self.lat], dtype=np.float32),
            _lon=np.array([self.lon], dtype=np.float32),
            _size_km=np.array([self.size_km], dtype=np.float32),
        )
        return path


class SimSatClient:
    """Thin requests-based client for the DPhi SimSat HTTP API."""

    def __init__(
        self,
        base_url: str = SIMSAT_BASE_URL,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- raw HTTP -------------------------------------------------------
    def _get(self, endpoint: str, params: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SimSatUnavailableError(
                f"Could not reach SimSat at {url}. Is `docker compose up` running? "
                f"Underlying error: {exc}"
            ) from exc
        if response.status_code != 200:
            raise SimSatRequestError(
                f"{endpoint} returned HTTP {response.status_code}: {response.text[:300]}"
            )
        return response

    # ---- typed helpers --------------------------------------------------
    def current_position(self) -> dict:
        """Return {'lon-lat-alt': [...], 'timestamp': '...'}."""
        return self._get("/data/current/position").json()

    def fetch_band_png(
        self,
        lat: float,
        lon: float,
        timestamp: str,
        spectral_band: str,
        size_km: float = 5.0,
    ) -> tuple[np.ndarray, dict]:
        """Fetch a single Sentinel-2 band as a 2D float32 array in [0, 1].

        SimSat returns a PNG; for a single-band request the three RGB
        channels are identical greyscale, so we collapse to one channel.
        """
        params = {
            "lon": lon,
            "lat": lat,
            "timestamp": timestamp,
            "spectral_bands": [spectral_band],
            "size_km": size_km,
            "return_type": "png",
        }
        response = self._get("/data/image/sentinel", params=params)

        try:
            metadata = json.loads(response.headers.get("sentinel_metadata", "{}"))
        except json.JSONDecodeError:
            metadata = {}
        if not metadata.get("image_available", False):
            raise SimSatNoImageError(
                f"SimSat reports no image available for band={spectral_band} "
                f"at ({lat:.4f}, {lon:.4f}) @ {timestamp}. "
                f"Try a different timestamp or location over land/daylight."
            )

        try:
            import matplotlib.image as mpimg
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to decode SimSat PNGs. "
                "Install it via `pip install matplotlib`."
            ) from exc

        image = mpimg.imread(io.BytesIO(response.content), format="PNG")
        # Greyscale collapse — single-band PNGs come back with R==G==B.
        if image.ndim == 3:
            image = image[..., :3].mean(axis=-1)
        return image.astype(np.float32), metadata

    def fetch_pathogen_scout_tile(
        self,
        lat: float,
        lon: float,
        timestamp: str,
        size_km: float = 5.0,
    ) -> SimSatTile:
        """Fetch B4 + B8 + B8A and assemble a Pathogen Scout-shaped tile."""
        bands: dict[str, np.ndarray] = {}
        last_meta: dict = {}
        for ps_band, simsat_band in PATHOGEN_SCOUT_BANDS.items():
            arr, meta = self.fetch_band_png(
                lat=lat, lon=lon, timestamp=timestamp,
                spectral_band=simsat_band, size_km=size_km,
            )
            bands[ps_band] = arr
            last_meta = meta

        # Sanity: all bands must have identical shape so NDRE math works.
        shapes = {k: v.shape for k, v in bands.items()}
        if len(set(shapes.values())) != 1:
            raise SimSatInconsistentShapeError(
                f"SimSat returned bands with inconsistent shapes: {shapes}"
            )

        return SimSatTile(
            bands=bands,
            lat=lat,
            lon=lon,
            timestamp=timestamp,
            size_km=size_km,
            metadata=last_meta,
        )


# --- Exceptions --------------------------------------------------------------
class SimSatError(RuntimeError):
    """Base class for SimSat client errors."""


class SimSatUnavailableError(SimSatError):
    """Raised when the SimSat HTTP server is unreachable."""


class SimSatRequestError(SimSatError):
    """Raised when SimSat returns a non-200 response."""


class SimSatNoImageError(SimSatError):
    """Raised when SimSat reports image_available=False for a request."""


class SimSatInconsistentShapeError(SimSatError):
    """Raised when fetched bands disagree on shape."""


# --- CLI ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a Pathogen Scout tile (B4, B8, B8A) from DPhi SimSat."
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Fetch at the satellite's current position (uses /data/current/position).",
    )
    parser.add_argument("--lat", type=float, help="Target latitude (degrees).")
    parser.add_argument("--lon", type=float, help="Target longitude (degrees).")
    parser.add_argument(
        "--timestamp",
        help="ISO-8601 UTC timestamp, e.g. 2026-03-01T16:00:00Z.",
    )
    parser.add_argument(
        "--size-km", type=float, default=5.0,
        help="Square tile side length in kilometers (default: 5).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output .npz path (consumed by agent_logic.py).",
    )
    parser.add_argument(
        "--base-url", default=SIMSAT_BASE_URL,
        help=f"SimSat API base URL (default: {SIMSAT_BASE_URL}).",
    )
    args = parser.parse_args()

    client = SimSatClient(base_url=args.base_url)

    if args.current:
        pos = client.current_position()
        lon, lat, _alt = pos["lon-lat-alt"]
        timestamp = pos["timestamp"]
        print(f"[simsat] satellite is at lat={lat:.4f}, lon={lon:.4f}, t={timestamp}")
    else:
        if args.lat is None or args.lon is None or args.timestamp is None:
            parser.error("--current is required, OR pass --lat, --lon, and --timestamp.")
        lat, lon, timestamp = args.lat, args.lon, args.timestamp

    print(f"[simsat] fetching B4 + B8 + B8A at ({lat:.4f}, {lon:.4f}) @ {timestamp} …")
    try:
        tile = client.fetch_pathogen_scout_tile(
            lat=lat, lon=lon, timestamp=timestamp, size_km=args.size_km,
        )
    except SimSatNoImageError as exc:
        print(f"[simsat] {exc}", file=sys.stderr)
        sys.exit(2)
    except SimSatUnavailableError as exc:
        print(f"[simsat] {exc}", file=sys.stderr)
        sys.exit(3)

    out = tile.save_npz(args.out)
    h, w = tile.bands["B4"].shape
    print(f"[simsat] saved {out} — shape=({h}, {w})  size_km={args.size_km}")
    print(f"[simsat] next step:  python -m src.agent_logic --tile {out} --verbose "
          f"--lat {lat} --lon {lon}")


if __name__ == "__main__":
    main()
