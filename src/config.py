"""Central configuration for the Pathogen Scout agent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "pathogen_classifier_int8.onnx"
OUTPUT_DIR = ROOT / "data" / "output_packets"

# Tier 1 — Sentinel
NDRE_STRESS_THRESHOLD = 0.20          # NDRE below this is "stressed"
ANOMALY_RATIO_TRIGGER = 0.02          # >=2% stressed pixels triggers Tier 2

# Tier 2 — Analyst
CLASSIFIER_INPUT_SIZE = (224, 224)
CLASS_LABELS = ["HEALTHY", "DROUGHT", "PATHOGEN"]
PATHOGEN_CONFIDENCE_FLOOR = 0.65      # min confidence to escalate to Tier 3

# Tier 3 — Dispatcher
SATELLITE_ID = "SimSat-Edge-01"
DRONE_RESPONSE_WINDOW = "T+02h"
