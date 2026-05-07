"""Smoke tests for the three-tier agent."""
from __future__ import annotations
import numpy as np
import tempfile
from pathlib import Path

from src.agent_logic import PathogenScoutAgent
from src.multispectral import compute_ndre, detect_stress


def _synthetic_tile(stress: bool, size: int = 256) -> dict[str, np.ndarray]:
    """Build a tile with healthy vegetation; optionally inject a stressed patch."""
    rng = np.random.default_rng(0)
    b4 = rng.uniform(0.05, 0.10, (size, size)).astype(np.float32)
    b8 = rng.uniform(0.40, 0.60, (size, size)).astype(np.float32)
    b8a = rng.uniform(0.45, 0.65, (size, size)).astype(np.float32)
    if stress:
        # Inject a patch where red-edge collapses toward red.
        b4[100:160, 100:160] = 0.30
        b8a[100:160, 100:160] = 0.32
    return {"B4": b4, "B8": b8, "B8A": b8a}


def test_ndre_range():
    bands = _synthetic_tile(stress=False)
    ndre = compute_ndre(bands["B4"], bands["B8A"])
    assert ndre.min() >= -1.0 and ndre.max() <= 1.0


def test_tier1_clears_healthy_tile(tmp_path: Path):
    bands = _synthetic_tile(stress=False)
    smap = detect_stress(bands, ndre_threshold=0.20)
    assert smap.anomaly_ratio < 0.02


def test_tier1_flags_stressed_tile():
    bands = _synthetic_tile(stress=True)
    smap = detect_stress(bands, ndre_threshold=0.20)
    assert smap.anomaly_ratio >= 0.02
    assert smap.centroid_yx is not None


def test_full_pipeline_on_stressed_tile(tmp_path: Path):
    bands = _synthetic_tile(stress=True)
    tile_path = tmp_path / "tile.npz"
    np.savez(tile_path, **bands)

    agent = PathogenScoutAgent()
    decision = agent.run(
        tile_path=tile_path,
        tile_origin_latlon=(28.7, 77.1),
        tile_pixel_size_deg=9e-5,
        tile_id="T43RGN",
    )
    assert decision.tier_reached >= 2
