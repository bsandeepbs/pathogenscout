"""Tier-3 Dispatcher — emits the Tactical Packet.

The Tactical Packet is the *only* artifact the satellite downlinks. It carries
the verdict, the AOI, and a drone-tasking command — never the source pixels.
This is the single mechanism by which Pathogen Scout achieves >99.9%
bandwidth reduction over conventional EO pipelines.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid


@dataclass
class TacticalPacket:
    packet_id: str
    satellite: str
    timestamp_utc: str
    geo: dict
    verdict: str
    confidence: float
    ndre_anomaly_ratio: float
    classifier_backend: str
    class_probabilities: dict
    drone_command: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def size_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))


def _pixel_to_geo(
    centroid_yx: tuple[int, int],
    tile_origin_latlon: tuple[float, float],
    tile_pixel_size_deg: float,
) -> tuple[float, float]:
    """Approximate pixel → lat/lon. Real systems use the tile's GeoTransform."""
    lat0, lon0 = tile_origin_latlon
    cy, cx = centroid_yx
    lat = lat0 - cy * tile_pixel_size_deg
    lon = lon0 + cx * tile_pixel_size_deg
    return round(lat, 6), round(lon, 6)


def build_packet(
    *,
    satellite_id: str,
    tile_id: str,
    tile_origin_latlon: tuple[float, float],
    tile_pixel_size_deg: float,
    centroid_yx: tuple[int, int],
    verdict: str,
    confidence: float,
    anomaly_ratio: float,
    backend: str,
    probs: dict,
    response_window: str,
) -> TacticalPacket:
    now = datetime.now(timezone.utc)
    lat, lon = _pixel_to_geo(centroid_yx, tile_origin_latlon, tile_pixel_size_deg)

    priority = "HIGH" if confidence >= 0.85 else "MEDIUM"
    drone_command = (
        f"TASK_DRONE :: PRIORITY={priority} "
        f":: AOI={lat},{lon} "
        f":: PAYLOAD=HYPERSPEC "
        f":: WINDOW={response_window}"
    )

    packet_id = (
        f"PS-{now.strftime('%Y%m%dT%H%MZ')}-{uuid.uuid4().hex[:4].upper()}"
    )

    return TacticalPacket(
        packet_id=packet_id,
        satellite=satellite_id,
        timestamp_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        geo={"lat": lat, "lon": lon, "tile_id": tile_id},
        verdict=verdict,
        confidence=round(confidence, 4),
        ndre_anomaly_ratio=round(anomaly_ratio, 4),
        classifier_backend=backend,
        class_probabilities={k: round(v, 4) for k, v in probs.items()},
        drone_command=drone_command,
    )


def write_packet(packet: TacticalPacket, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"packet_{packet.packet_id}.json"
    out.write_text(packet.to_json(), encoding="utf-8")
    return out
