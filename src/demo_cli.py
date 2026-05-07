"""Rich-formatted streaming CLI — the "flight software" demo.

Scans a directory of tiles, runs each through the three-tier agent, and
prints a continuously-updating mission log: the kind of output you'd
see piped from a real on-board process. Designed to be screen-recorded
into a 15-second GIF for the README.

    python -m src.demo_cli --tiles data/sample_tiles/

If the directory is empty, the demo synthesizes a deterministic mix of
healthy / drought / pathogen tiles so you can run it on a fresh clone.
"""
from __future__ import annotations
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from src import config
from src.agent_logic import PathogenScoutAgent

# Windows' default console codec (cp1252) cannot render the emoji we use.
# Reconfigure stdout to UTF-8 so Rich panels render identically across OSes.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

console = Console(force_terminal=True, legacy_windows=False)

VERDICT_STYLE = {
    "CLEAR": "dim green",
    "HEALTHY_LOGGED": "green",
    "DROUGHT_LOGGED": "yellow",
    "PATHOGEN_SUSPECTED": "bold red",
}


def _synth_tile(kind: str, size: int = 256) -> dict[str, np.ndarray]:
    """Deterministic synthetic tiles for demo without real data.

    Calibrated so each kind triggers a different agent path:
      healthy  → Tier 1 clears the tile.
      drought  → Tier 1 trips, Tier 2 classifies as DROUGHT (uniform red-edge depression).
      pathogen → Tier 1 trips, Tier 2 classifies as PATHOGEN (patchy variance).
    """
    rng = np.random.default_rng({"healthy": 1, "drought": 2, "pathogen": 3}[kind])
    b4 = rng.uniform(0.05, 0.10, (size, size)).astype(np.float32)
    b8 = rng.uniform(0.40, 0.60, (size, size)).astype(np.float32)
    b8a = rng.uniform(0.45, 0.65, (size, size)).astype(np.float32)

    if kind == "drought":
        # Uniform red-edge depression — broad but homogeneous stress.
        b4 += 0.18
        b8a -= 0.20
    elif kind == "pathogen":
        # Patchy red-edge collapse — heterogeneous, high local variance.
        # Eight ~40×40 lesions ≈ 19% anomaly ratio → comfortably above 2% trigger.
        patches = [(20, 30), (40, 150), (90, 60), (90, 180),
                   (140, 100), (170, 30), (190, 180), (220, 110)]
        for cy, cx in patches:
            y0, y1 = cy, min(cy + 40, size)
            x0, x1 = cx, min(cx + 40, size)
            b4[y0:y1, x0:x1] = 0.32
            b8a[y0:y1, x0:x1] = 0.30
    return {"B4": b4, "B8": b8, "B8A": b8a}


def _materialize_demo_tiles(target_dir: Path) -> list[Path]:
    """Build a deterministic 8-tile mix on disk for the demo."""
    target_dir.mkdir(parents=True, exist_ok=True)
    plan = [
        ("tile_001_healthy",  "healthy"),
        ("tile_002_healthy",  "healthy"),
        ("tile_003_drought",  "drought"),
        ("tile_004_healthy",  "healthy"),
        ("tile_005_pathogen", "pathogen"),
        ("tile_006_healthy",  "healthy"),
        ("tile_007_drought",  "drought"),
        ("tile_008_pathogen", "pathogen"),
    ]
    paths = []
    for name, kind in plan:
        p = target_dir / f"{name}.npz"
        if not p.exists():
            np.savez(p, **_synth_tile(kind))
        paths.append(p)
    return paths


def _format_kb(b: int) -> str:
    return f"{b/1024:.2f} KB" if b < 1_000_000 else f"{b/1_048_576:.2f} MB"


def _emit_log_line(idx: int, total: int, tile: Path, decision) -> None:
    style = VERDICT_STYLE.get(decision.verdict, "white")
    saved_pct = 0.0
    if decision.tile_bytes:
        saved_pct = 100 * (1 - decision.packet_bytes / decision.tile_bytes) \
            if decision.packet_bytes else 100.0

    header = Text()
    header.append(f"[{idx:02d}/{total:02d}] ", style="dim")
    header.append("ORBIT-PASS ", style="bold cyan")
    header.append(f"{tile.name:<28}", style="white")
    header.append(" → ")
    header.append(f"T{decision.tier_reached} ", style="bold magenta")
    header.append(decision.verdict, style=style)

    detail = Text()
    detail.append("    └─ ", style="dim")
    detail.append(f"tile={_format_kb(decision.tile_bytes)}", style="dim")
    detail.append("  ")
    if decision.packet_bytes:
        detail.append(f"packet={_format_kb(decision.packet_bytes)}", style="bright_white")
        detail.append("  ")
        detail.append(f"saved={saved_pct:.4f}%", style="bold green")
        detail.append("  ")
        detail.append(f"latency={decision.elapsed_ms:.1f}ms", style="dim")
    else:
        detail.append("packet=∅ (no downlink)", style="dim cyan")
        detail.append("  ")
        detail.append(f"latency={decision.elapsed_ms:.1f}ms", style="dim")

    console.print(header)
    console.print(detail)


def run_demo(tile_dir: Path, slow: float = 0.0) -> None:
    if not tile_dir.exists() or not list(tile_dir.glob("*.npz")):
        console.print(Panel.fit(
            "[yellow]No tiles found — synthesizing 8 demo tiles…[/yellow]",
            border_style="yellow",
        ))
        tiles = _materialize_demo_tiles(tile_dir)
    else:
        tiles = sorted(tile_dir.glob("*.npz"))

    console.print()
    console.print(Panel.fit(
        Text.from_markup(
            "[bold cyan]🛰  PATHOGEN SCOUT[/bold cyan]  ::  "
            "[white]SimSat-Edge-01[/white]  ::  "
            f"[dim]orbit pass · {len(tiles)} tiles queued[/dim]"
        ),
        border_style="cyan",
    ))
    console.print()

    agent = PathogenScoutAgent()

    total_tile_bytes = 0
    total_packet_bytes = 0
    verdicts: dict[str, int] = {}

    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]processing orbit pass"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task("scan", total=len(tiles))

    with progress:
        for idx, tile in enumerate(tiles, start=1):
            decision = agent.run(
                tile_path=tile,
                tile_origin_latlon=(45.20, 12.80),
                tile_pixel_size_deg=9e-5,
                tile_id=tile.stem.upper(),
            )
            _emit_log_line(idx, len(tiles), tile, decision)
            total_tile_bytes += decision.tile_bytes
            total_packet_bytes += decision.packet_bytes
            verdicts[decision.verdict] = verdicts.get(decision.verdict, 0) + 1
            progress.update(task, advance=1)
            if slow:
                time.sleep(slow)

    # ---- Summary panel ---------------------------------------------------
    console.print()
    table = Table(
        title="[bold]Mission Summary — Downlink Budget[/bold]",
        title_justify="left",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Metric", style="white")
    table.add_column("Conventional", justify="right", style="red")
    table.add_column("Pathogen Scout", justify="right", style="green")
    table.add_column("Δ", justify="right", style="bold")

    saved = total_tile_bytes - total_packet_bytes
    ratio = total_tile_bytes / max(total_packet_bytes, 1) if total_packet_bytes else float("inf")
    saved_pct = 100 * saved / max(total_tile_bytes, 1)

    table.add_row(
        "Bytes downlinked",
        _format_kb(total_tile_bytes),
        _format_kb(total_packet_bytes) if total_packet_bytes else "0 B",
        f"−{_format_kb(saved)}",
    )
    table.add_row(
        "Compression",
        "1×",
        f"{ratio:,.0f}×" if total_packet_bytes else "∞",
        f"[bold green]{saved_pct:.4f}% saved[/bold green]",
    )
    table.add_row(
        "Tiles processed",
        f"{len(tiles)}",
        f"{len(tiles)}",
        "—",
    )

    verdict_summary = "  ".join(
        f"[{VERDICT_STYLE.get(v, 'white')}]{v}[/]: {n}" for v, n in verdicts.items()
    )

    console.print(table)
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold]Verdicts:[/bold]   {verdict_summary}\n"
            f"[bold]Packets emitted:[/bold]  {verdicts.get('PATHOGEN_SUSPECTED', 0)}  "
            f"→  written to [cyan]{config.OUTPUT_DIR}[/cyan]"
        ),
        border_style="green",
        title="[bold green]ORBIT PASS COMPLETE[/bold green]",
        title_align="left",
    ))
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathogen Scout — streaming demo.")
    parser.add_argument("--tiles", type=Path, default=config.ROOT / "data" / "sample_tiles")
    parser.add_argument("--slow", type=float, default=0.35,
                        help="Seconds to pause between tiles (set 0 to disable).")
    args = parser.parse_args()
    run_demo(args.tiles, slow=args.slow)


if __name__ == "__main__":
    main()
