"""Pathogen Scout — agent orchestrator.

Three-tier escalation:
  Tier 1  (Sentinel)   →  cheap NumPy NDRE pass.            Discards 95%+ of tiles.
  Tier 2  (Analyst)    →  INT8 MobileNetV3 over ONNX.       Discards drought / healthy.
  Tier 3  (Dispatcher) →  emits ~2 KB Tactical Packet JSON.

Run:
    python -m src.agent_logic --tile data/sample_tiles/demo_tile.npz \
                              --lat 28.7 --lon 77.1 --tile-id T43RGN
"""
from __future__ import annotations
import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.classifier import PathogenClassifier
from src.dispatcher import build_packet, write_packet
from src.multispectral import (
    StressMap,
    crop_roi,
    detect_stress,
    load_tile,
)

log = logging.getLogger("pathogen_scout")


@dataclass
class AgentDecision:
    tier_reached: int
    verdict: str                          # "CLEAR", "DROUGHT_LOGGED", "PATHOGEN_SUSPECTED"
    packet_path: Path | None = None
    packet_bytes: int = 0
    tile_bytes: int = 0
    elapsed_ms: float = 0.0


class PathogenScoutAgent:
    """Orchestrates Tier 1 → Tier 3 for a single tile."""

    def __init__(self):
        self.classifier = PathogenClassifier(
            model_path=config.MODEL_PATH,
            labels=config.CLASS_LABELS,
        )

    # --- Tier 1 ----------------------------------------------------------
    def tier1_sentinel(self, bands: dict[str, np.ndarray]) -> StressMap:
        log.info("[T1] Sentinel scanning tile (NDRE)…")
        smap = detect_stress(bands, ndre_threshold=config.NDRE_STRESS_THRESHOLD)
        log.info(
            "[T1] anomaly_ratio=%.4f  (trigger=%.2f)",
            smap.anomaly_ratio, config.ANOMALY_RATIO_TRIGGER,
        )
        return smap

    # --- Tier 2 ----------------------------------------------------------
    def tier2_analyst(
        self, bands: dict[str, np.ndarray], smap: StressMap,
    ):
        log.info("[T2] Analyst classifying ROI (ONNX backend=%s)…",
                 "available" if self.classifier.session else "heuristic")
        roi = crop_roi(
            bands, smap.centroid_yx, size=config.CLASSIFIER_INPUT_SIZE,
        )
        # Extract un-normalized NDRE patch around the centroid for the
        # heuristic fallback (the ONNX backend ignores this).
        cy, cx = smap.centroid_yx
        h, w = smap.ndre.shape
        half = 64
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        ndre_patch = smap.ndre[y0:y1, x0:x1]

        result = self.classifier.predict(roi, ndre_patch=ndre_patch)
        log.info("[T2] verdict=%s  confidence=%.3f", result.label, result.confidence)
        return result

    # --- Tier 3 ----------------------------------------------------------
    def tier3_dispatcher(
        self,
        smap: StressMap,
        classification,
        tile_id: str,
        tile_origin_latlon: tuple[float, float],
        tile_pixel_size_deg: float,
    ):
        log.info("[T3] Dispatcher building Tactical Packet…")
        packet = build_packet(
            satellite_id=config.SATELLITE_ID,
            tile_id=tile_id,
            tile_origin_latlon=tile_origin_latlon,
            tile_pixel_size_deg=tile_pixel_size_deg,
            centroid_yx=smap.centroid_yx,
            verdict="PATHOGEN_SUSPECTED",
            confidence=classification.confidence,
            anomaly_ratio=smap.anomaly_ratio,
            backend=classification.backend,
            probs=classification.probs,
            response_window=config.DRONE_RESPONSE_WINDOW,
        )
        out_path = write_packet(packet, config.OUTPUT_DIR)
        log.info("[T3] packet=%s  bytes=%d", out_path.name, packet.size_bytes())
        return packet, out_path

    # --- Pipeline --------------------------------------------------------
    def run(
        self,
        tile_path: Path,
        tile_origin_latlon: tuple[float, float],
        tile_pixel_size_deg: float,
        tile_id: str,
    ) -> AgentDecision:
        t0 = time.perf_counter()
        bands = load_tile(tile_path)
        tile_bytes = sum(b.nbytes for b in bands.values())

        # Tier 1
        smap = self.tier1_sentinel(bands)
        if (smap.anomaly_ratio < config.ANOMALY_RATIO_TRIGGER
                or smap.centroid_yx is None):
            return AgentDecision(
                tier_reached=1,
                verdict="CLEAR",
                tile_bytes=tile_bytes,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Tier 2
        classification = self.tier2_analyst(bands, smap)
        if (classification.label != "PATHOGEN"
                or classification.confidence < config.PATHOGEN_CONFIDENCE_FLOOR):
            return AgentDecision(
                tier_reached=2,
                verdict=f"{classification.label}_LOGGED",
                tile_bytes=tile_bytes,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # Tier 3
        packet, packet_path = self.tier3_dispatcher(
            smap, classification, tile_id,
            tile_origin_latlon, tile_pixel_size_deg,
        )
        return AgentDecision(
            tier_reached=3,
            verdict="PATHOGEN_SUSPECTED",
            packet_path=packet_path,
            packet_bytes=packet.size_bytes(),
            tile_bytes=tile_bytes,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


def _print_summary(decision: AgentDecision) -> None:
    print("\n" + "=" * 60)
    print(f"  Pathogen Scout — Decision Report")
    print("=" * 60)
    print(f"  Tier reached     : {decision.tier_reached}")
    print(f"  Verdict          : {decision.verdict}")
    print(f"  Elapsed          : {decision.elapsed_ms:.1f} ms")
    if decision.packet_path:
        ratio = decision.tile_bytes / max(decision.packet_bytes, 1)
        saved = 100 * (1 - decision.packet_bytes / max(decision.tile_bytes, 1))
        print(f"  Tile size        : {decision.tile_bytes:,} bytes")
        print(f"  Packet size      : {decision.packet_bytes:,} bytes")
        print(f"  Compression      : {ratio:,.0f}× ({saved:.4f}% downlink saved)")
        print(f"  Packet written   : {decision.packet_path}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathogen Scout edge agent.")
    parser.add_argument("--tile", required=True, type=Path)
    parser.add_argument("--lat", type=float, default=28.7041,
                        help="Latitude of the tile's top-left pixel.")
    parser.add_argument("--lon", type=float, default=77.1025,
                        help="Longitude of the tile's top-left pixel.")
    parser.add_argument("--pixel-size-deg", type=float, default=9e-5,
                        help="Approx degrees per pixel (~10 m at equator).")
    parser.add_argument("--tile-id", default="T43RGN")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = PathogenScoutAgent()
    decision = agent.run(
        tile_path=args.tile,
        tile_origin_latlon=(args.lat, args.lon),
        tile_pixel_size_deg=args.pixel_size_deg,
        tile_id=args.tile_id,
    )
    _print_summary(decision)


if __name__ == "__main__":
    main()
