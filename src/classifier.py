"""Tier-2 Analyst — INT8-quantized MobileNetV3 over ONNX Runtime.

The model is trained off-line on a corpus of {Healthy, Drought, Pathogen}
red-edge crops and exported with dynamic INT8 quantization. On a Coral-class
edge accelerator this runs in ~30–50 ms per ROI at <1 W.

If the .onnx file is not present (e.g. during a hackathon dry-run), we fall
back to a deterministic NDRE-statistics heuristic so the agent pipeline
remains end-to-end runnable.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class Classification:
    label: str
    confidence: float
    probs: dict[str, float]
    backend: str  # "onnx-int8" or "heuristic"


class PathogenClassifier:
    def __init__(self, model_path: Path, labels: list[str]):
        self.labels = labels
        self.session = None
        self.input_name = None
        if ort is not None and Path(model_path).exists():
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 1  # edge: single core, deterministic
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        roi: np.ndarray,
        ndre_patch: np.ndarray | None = None,
    ) -> Classification:
        """Classify the ROI.

        Args:
            roi: HxWxC float32 in [0, 1] — ready for the ONNX model.
            ndre_patch: optional un-normalized NDRE values from the original
                tile (same spatial region as roi). Used by the heuristic
                fallback to recover absolute NDRE scale lost during
                min-max normalization. Ignored when ONNX is available.
        """
        if self.session is not None:
            tensor = roi.transpose(2, 0, 1)[None, ...].astype(np.float32)
            logits = self.session.run(None, {self.input_name: tensor})[0][0]
            probs = _softmax(logits)
            backend = "onnx-int8"
        else:
            probs = self._heuristic(roi, ndre_patch)
            backend = "heuristic"

        idx = int(np.argmax(probs))
        return Classification(
            label=self.labels[idx],
            confidence=float(probs[idx]),
            probs={lbl: float(p) for lbl, p in zip(self.labels, probs)},
            backend=backend,
        )

    def _heuristic(
        self,
        roi: np.ndarray,
        ndre_patch: np.ndarray | None,
    ) -> np.ndarray:
        """Fallback: separate drought from pathogen via red-edge texture.

        Agronomic discriminators:
          • Pathogens → *patchy* red-edge collapse: bimodal NDRE,
            high spatial variance, mixed stressed + healthy pixels.
          • Drought   → *uniform* red-edge depression: unimodal NDRE,
            low variance, the whole ROI is depressed.
          • Healthy   → high mean NDRE with low variance.

        Uses the un-normalized NDRE patch when available (preserves absolute
        scale); falls back to ROI-derived NDRE otherwise.
        """
        if ndre_patch is not None and ndre_patch.size:
            ndre = ndre_patch
        else:
            red, _nir, redge = roi[..., 0], roi[..., 1], roi[..., 2]
            ndre = (redge - red) / np.maximum(redge + red, 1e-6)

        mean_ndre = float(ndre.mean())
        ndre_std = float(ndre.std())
        stressed_frac = float((ndre < 0.20).mean())

        # Logits — high mean NDRE → healthy; high stressed_frac with low std → drought;
        # high stressed_frac with high std → pathogen (mix of healthy + collapsed pixels).
        healthy_logit  = 6.0 * mean_ndre - 1.0
        drought_logit  = 5.0 * stressed_frac - 12.0 * ndre_std + 0.5
        pathogen_logit = 3.0 * stressed_frac + 14.0 * ndre_std - 1.0
        return _softmax(np.array([healthy_logit, drought_logit, pathogen_logit]))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
