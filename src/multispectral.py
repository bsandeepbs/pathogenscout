"""Multispectral Engine — NDRE computation on Sentinel-2 bands.

NDRE = (B8A - B4) / (B8A + B4)

NDRE leverages the red-edge band (B8A, ~865 nm) instead of NIR (B8). It is
significantly more sensitive to mid-to-late-stage chlorophyll degradation,
which is the signature of pathogen-induced stress *before* it becomes
visible in true-colour imagery.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2


@dataclass
class StressMap:
    ndre: np.ndarray              # float32, shape (H, W), range [-1, 1]
    stress_mask: np.ndarray       # uint8,   shape (H, W), {0, 1}
    anomaly_ratio: float          # fraction of stressed pixels
    centroid_yx: tuple[int, int] | None  # row, col of stress centroid


def load_tile(path: str | Path) -> dict[str, np.ndarray]:
    """Load a Sentinel-2 tile saved as a .npz with keys B4, B8, B8A."""
    arr = np.load(path)
    required = {"B4", "B8", "B8A"}
    missing = required - set(arr.files)
    if missing:
        raise ValueError(f"Tile missing required bands: {missing}")
    return {k: arr[k].astype(np.float32) for k in required}


def compute_ndre(b4: np.ndarray, b8a: np.ndarray) -> np.ndarray:
    """Normalized Difference Red Edge index. Safe against div-by-zero."""
    num = b8a - b4
    den = b8a + b4
    ndre = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return ndre.astype(np.float32)


def detect_stress(
    bands: dict[str, np.ndarray],
    ndre_threshold: float,
) -> StressMap:
    """Tier-1 sentinel pass — pure NumPy/OpenCV, no ML."""
    ndre = compute_ndre(bands["B4"], bands["B8A"])

    # Vegetation gate: only consider pixels that are actually vegetated.
    # NDVI < 0.2 is bare soil / water — exclude from stress accounting.
    ndvi = compute_ndre(bands["B4"], bands["B8"])  # same formula structure
    veg_gate = ndvi > 0.2

    raw_mask = (ndre < ndre_threshold) & veg_gate
    mask = raw_mask.astype(np.uint8)

    # Morphological open removes salt-and-pepper false positives.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    veg_pixels = int(veg_gate.sum())
    anomaly_ratio = float(mask.sum()) / max(veg_pixels, 1)

    centroid = None
    if mask.any():
        ys, xs = np.where(mask > 0)
        centroid = (int(ys.mean()), int(xs.mean()))

    return StressMap(
        ndre=ndre,
        stress_mask=mask,
        anomaly_ratio=anomaly_ratio,
        centroid_yx=centroid,
    )


def crop_roi(
    bands: dict[str, np.ndarray],
    centroid_yx: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
    """Extract a 3-channel (B4, B8, B8A) ROI for the classifier, resized."""
    h, w = bands["B4"].shape
    cy, cx = centroid_yx
    half_h, half_w = size[0] // 2, size[1] // 2
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)

    stack = np.stack(
        [bands["B4"][y0:y1, x0:x1],
         bands["B8"][y0:y1, x0:x1],
         bands["B8A"][y0:y1, x0:x1]],
        axis=-1,
    )
    # Min-max normalize per ROI to [0, 1] for stable inference.
    lo, hi = stack.min(), stack.max()
    stack = (stack - lo) / max(hi - lo, 1e-6)
    roi = cv2.resize(stack, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return roi.astype(np.float32)
