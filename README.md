# 🛰️ Pathogen Scout

> **Agentic on-board satellite intelligence for pre-visual crop biosecurity threat detection.**
> Built for the **DPhi SimSat Hackathon — General AI Track**.

---

## 🌍 The Problem

Traditional Earth-observation satellites operate on a brittle pipeline:

1. Capture massive multispectral tiles (~1 GB / scene).
2. Store on board until a ground-station pass (often 6–12 hours later).
3. Downlink **everything**, regardless of value.
4. Process on the ground, then alert farmers — *days* after the fact.

For crop pathogens (e.g., wheat rust, citrus greening), **48 hours of latency = thousands of hectares lost.** And the real bottleneck isn't compute — it's **bandwidth**. A LEO satellite has roughly **10 minutes of downlink per orbit**.

## 🎯 Value Proposition — Why This Belongs In Orbit

**Pathogen Scout flips the model: think on-orbit, transmit only decisions.**

| Conventional Pipeline | Pathogen Scout (Agentic Edge) |
|-----------------------|-------------------------------|
| Downlink ~1 GB raw tile | Downlink ~2 KB JSON Tactical Packet |
| Latency: 6–48 hours | Latency: < 90 seconds |
| Ground-side analysis | On-board three-tier agent |
| Captures everything, decides later | **Decides in orbit, captures attention only when it matters** |

### 📉 Bandwidth Savings: ~99.9998% reduction in data transmission

A 1024×1024×4-band uint16 tile is ≈ **8 MB** compressed. A Tactical Packet — GPS coordinates, disease probability, classifier confidence, and a drone-command string — is **under 2 KB**. That is a **>4000× reduction**, freeing the downlink budget for the *truly* anomalous events.

### 🚀 Why Edge Compute Is Not Optional In Space

- **Power budget:** A 6U CubeSat has ≈ 20 W peak. Our Tier 1 sentinel runs on integer NDRE math (NumPy/OpenCV) — no GPU required.
- **Thermal limits:** Sustained inference cooks radiators. INT8-quantized MobileNetV3 over ONNX Runtime keeps inference under 50 ms/tile on a Coral-class accelerator.
- **No human in the loop:** During the 80% of an orbit spent over ocean or out-of-contact, the satellite must reason **autonomously**. Our three-tier gate models exactly that — escalate cheap → expensive → actionable.
- **Decision-grade output:** Ground operators don't need pixels — they need a coordinate and a verdict. The Dispatcher emits a drone-tasking command, not a postcard.

> **Pathogen Scout is not a model. It is an agentic policy for scarce orbits, scarce watts, and scarce bandwidth.**

---

## 🧠 Architecture — The Three-Tier Agent

```
                    ┌────────────────────────────────┐
                    │   RAW MULTISPECTRAL TILE       │
                    │   (B4 Red │ B8 NIR │ B8A RE)   │
                    └────────────────┬───────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────┐
        │  TIER 1 · THE SENTINEL  (NumPy / OpenCV)        │
        │  • Compute NDRE = (B8A - B4) / (B8A + B4)       │
        │  • Flag pixels where NDRE < threshold           │
        │  • If anomaly_ratio < 2%  →  DISCARD            │
        └────────────────┬────────────────────────────────┘
                         ▼ (anomaly detected)
        ┌─────────────────────────────────────────────────┐
        │  TIER 2 · THE ANALYST  (ONNX INT8 MobileNetV3)  │
        │  • Crop ROI around anomaly                      │
        │  • Classify: {Healthy, Drought, Pathogen}       │
        │  • If class != Pathogen  →  LOG & DISCARD       │
        └────────────────┬────────────────────────────────┘
                         ▼ (pathogen suspected)
        ┌─────────────────────────────────────────────────┐
        │  TIER 3 · THE DISPATCHER                        │
        │  • Build Tactical Packet (~2 KB JSON)           │
        │  • Emit drone-tasking command                   │
        │  • Queue for next downlink window               │
        └─────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
pathogenscout/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agent_logic.py          # Orchestrates Tier 1 → Tier 3
│   ├── multispectral.py        # NDRE computation (NumPy/OpenCV)
│   ├── classifier.py           # ONNX INT8 MobileNetV3 wrapper
│   ├── dispatcher.py           # Tactical Packet builder
│   └── config.py               # Thresholds, paths, constants
├── models/
│   └── pathogen_classifier_int8.onnx   # Quantized weights
├── notebooks/
│   └── 01_ndre_exploration.ipynb
├── data/
│   ├── sample_tiles/           # Example Sentinel-2 inputs
│   └── output_packets/         # Generated Tactical Packets
└── tests/
    └── test_agent_logic.py
```

---

## ⚙️ Quickstart

```bash
pip install -r requirements.txt
python -m src.agent_logic --tile data/sample_tiles/demo_tile.npz
```

Output is written to `data/output_packets/packet_<timestamp>.json`.

---

## 📦 Tactical Packet Schema

```json
{
  "packet_id": "PS-20260506T1432Z-0001",
  "satellite": "SimSat-Edge-01",
  "timestamp_utc": "2026-05-06T14:32:18Z",
  "geo": { "lat": 28.6139, "lon": 77.2090, "tile_id": "T43RGN" },
  "verdict": "PATHOGEN_SUSPECTED",
  "confidence": 0.91,
  "ndre_anomaly_ratio": 0.073,
  "drone_command": "TASK_DRONE :: PRIORITY=HIGH :: AOI=28.6139,77.2090 :: PAYLOAD=HYPERSPEC :: WINDOW=T+02h"
}
```

---

## 🏆 Hackathon Track

DPhi SimSat — **General AI Track** · Theme: *Autonomous on-board reasoning for biosecurity.*
