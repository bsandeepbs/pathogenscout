"""Mission Control dashboard — Streamlit UI over the Pathogen Scout agent.

Run:
    streamlit run src/dashboard.py

Three-column view (Raw RGB | NDRE Stress Map | Agent Log) with a bandwidth
meter at the top showing the live downlink-saved ratio for the selected tile.
"""
from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st

from src import config
from src.agent_logic import PathogenScoutAgent
from src.demo_cli import _materialize_demo_tiles, _synth_tile
from src.multispectral import compute_ndre, detect_stress, load_tile

st.set_page_config(
    page_title="Pathogen Scout — Mission Control",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="font-size:2.4rem;">🛰️</div>
      <div>
        <div style="font-size:1.6rem;font-weight:700;line-height:1;">Pathogen Scout — Mission Control</div>
        <div style="color:#888;font-size:0.95rem;">SimSat-Edge-01 · agentic on-board biosecurity intelligence</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — tile selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📡 Orbit Pass Input")

    sample_dir = config.ROOT / "data" / "sample_tiles"
    if not list(sample_dir.glob("*.npz")):
        st.info("No sample tiles found — generating 8 demo tiles…")
        _materialize_demo_tiles(sample_dir)

    available = sorted(p.name for p in sample_dir.glob("*.npz"))
    chosen = st.selectbox("Select a tile from this orbit pass", available, index=0)

    st.markdown("---")
    uploaded = st.file_uploader(
        "…or upload your own .npz tile (keys: B4, B8, B8A)", type=["npz"]
    )

    st.markdown("---")
    st.caption("**Tier-1 NDRE threshold**")
    ndre_thresh = st.slider(
        "Stressed if NDRE <", -0.2, 0.6, config.NDRE_STRESS_THRESHOLD, 0.01
    )
    st.caption("**Pathogen confidence floor**")
    conf_floor = st.slider(
        "Escalate to Tier 3 if confidence ≥", 0.0, 1.0,
        config.PATHOGEN_CONFIDENCE_FLOOR, 0.01,
    )

# ---------------------------------------------------------------------------
# Load the chosen tile
# ---------------------------------------------------------------------------
if uploaded is not None:
    arr = np.load(io.BytesIO(uploaded.read()))
    bands = {k: arr[k].astype(np.float32) for k in ("B4", "B8", "B8A")}
    tile_label = uploaded.name
    tile_path = None
else:
    tile_path = sample_dir / chosen
    bands = load_tile(tile_path)
    tile_label = chosen


# ---------------------------------------------------------------------------
# Run pipeline (with overrides applied via temporary config patch)
# ---------------------------------------------------------------------------
config.NDRE_STRESS_THRESHOLD = ndre_thresh
config.PATHOGEN_CONFIDENCE_FLOOR = conf_floor

agent = PathogenScoutAgent()
if tile_path is None:
    # Persist uploaded tile so agent can re-load it.
    tmp = sample_dir / f"_uploaded_{tile_label}"
    np.savez(tmp, **bands)
    tile_path = tmp

decision = agent.run(
    tile_path=tile_path,
    tile_origin_latlon=(45.20, 12.80),
    tile_pixel_size_deg=9e-5,
    tile_id=Path(tile_label).stem.upper(),
)

# Re-derive the stress map for visualization (cheap).
smap = detect_stress(bands, ndre_threshold=ndre_thresh)

# ---------------------------------------------------------------------------
# Bandwidth meter (the headline)
# ---------------------------------------------------------------------------
tile_kb = decision.tile_bytes / 1024
packet_kb = decision.packet_bytes / 1024
saved_kb = max(tile_kb - packet_kb, 0)
saved_pct = 100 * saved_kb / tile_kb if tile_kb else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tile size (raw)", f"{tile_kb:,.1f} KB")
m2.metric(
    "Packet downlinked",
    f"{packet_kb:,.2f} KB" if packet_kb else "0 KB",
    delta=("emitted" if packet_kb else "discarded on-board"),
    delta_color=("normal" if packet_kb else "off"),
)
m3.metric(
    "Bandwidth saved",
    f"{saved_pct:.4f} %",
    delta=f"−{saved_kb:,.1f} KB",
)
m4.metric("Tier reached", f"T{decision.tier_reached}", delta=decision.verdict)

# Visual progress bar version of the savings.
st.progress(min(saved_pct / 100, 1.0), text=f"Downlink budget freed for higher-priority orbit traffic — {saved_pct:.4f}% saved")

st.divider()

# ---------------------------------------------------------------------------
# Three-column workspace
# ---------------------------------------------------------------------------
col_raw, col_ndre, col_log = st.columns([1, 1, 1.2])


def _to_rgb(bands: dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.stack([bands["B4"], bands["B8"], bands["B8A"]], axis=-1)
    lo, hi = rgb.min(), rgb.max()
    rgb = (rgb - lo) / max(hi - lo, 1e-6)
    return (rgb * 255).astype(np.uint8)


with col_raw:
    st.markdown("#### 📷 Raw Multispectral")
    st.caption("Channels: R=B4 · G=B8 · B=B8A")
    st.image(_to_rgb(bands), use_container_width=True)

with col_ndre:
    st.markdown("#### 🌿 NDRE Stress Map")
    st.caption("Red = stressed pixels (NDRE below threshold, vegetation only)")
    overlay = _to_rgb(bands).copy()
    mask = smap.stress_mask.astype(bool)
    overlay[mask] = [255, 40, 40]
    st.image(overlay, use_container_width=True)
    st.metric("Anomaly ratio", f"{smap.anomaly_ratio*100:.2f} %")

with col_log:
    st.markdown("#### 🧠 Agent Log")
    log_lines = [
        f"`[T1]` Sentinel scan complete — anomaly_ratio = **{smap.anomaly_ratio:.4f}** "
        f"(trigger ≥ {config.ANOMALY_RATIO_TRIGGER:.2f})",
    ]
    if decision.tier_reached >= 2:
        log_lines.append(
            f"`[T2]` Analyst classifier escalated — "
            f"reached tier {decision.tier_reached}"
        )
    if decision.tier_reached == 3:
        log_lines.append(
            f"`[T3]` 🚨 **PATHOGEN_SUSPECTED** — Tactical Packet emitted "
            f"({decision.packet_bytes:,} bytes)"
        )
    elif decision.tier_reached == 2:
        log_lines.append(
            f"`[T3]` ⏭️  Verdict `{decision.verdict}` — no downlink, no drone tasking"
        )
    else:
        log_lines.append("`[T2/T3]` ⏭️  Tile cleared at Tier 1 — discarded on-board")

    log_lines.append(f"`[⏱]` Total elapsed: **{decision.elapsed_ms:.1f} ms**")

    for line in log_lines:
        st.markdown("- " + line)

    if decision.packet_path:
        with open(decision.packet_path, "r", encoding="utf-8") as fh:
            packet_json = json.load(fh)
        st.markdown("##### 📦 Tactical Packet (downlinked)")
        st.json(packet_json, expanded=False)
        st.download_button(
            "⬇️ Download Tactical Packet",
            data=json.dumps(packet_json, indent=2),
            file_name=decision.packet_path.name,
            mime="application/json",
        )
    else:
        st.info("No packet emitted — agent decided this tile is not worth downlinking.")
