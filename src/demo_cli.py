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


def _try_fetch_simsat_tiles(target_dir: Path, n: int = 4) -> list[Path] | None:
    """If a local SimSat instance is up, pull `n` live tiles into target_dir.
    Returns the list of saved paths on success, or None if SimSat is
    unreachable / not configured (so the caller falls back to synthetic).
    """
    try:
        from src.simsat_client import (
            SimSatClient,
            SimSatNoImageError,
            SimSatUnavailableError,
        )
    except ImportError:
        return None

    client = SimSatClient()
    try:
        # Probe — the position endpoint is cheap and tells us SimSat is live.
        pos = client.current_position()
    except SimSatUnavailableError:
        return None
    except Exception as exc:
        console.print(f"[yellow][simsat] probe failed: {exc}[/yellow]")
        return None

    console.print(Panel.fit(
        Text.from_markup(
            f"[bold green]✓[/bold green] [white]SimSat live[/white] at "
            f"[cyan]{client.base_url}[/cyan] — fetching {n} real Sentinel-2 tiles "
            f"[dim](B4 / B8 / B8A)[/dim]"
        ),
        border_style="green",
    ))

    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    lon, lat, _alt = pos["lon-lat-alt"]
    timestamp = pos["timestamp"]
    for i in range(n):
        # Walk a tiny grid around the satellite ground track so each tile
        # is a *distinct* fetch but still in the current visibility window.
        d_lat = lat + (i - n / 2) * 0.05
        d_lon = lon + (i - n / 2) * 0.05
        out = target_dir / f"simsat_{i:02d}_{d_lat:.3f}_{d_lon:.3f}.npz"
        try:
            tile = client.fetch_pathogen_scout_tile(
                lat=d_lat, lon=d_lon, timestamp=timestamp,
            )
            tile.save_npz(out)
            saved.append(out)
        except SimSatNoImageError as exc:
            console.print(f"[yellow][simsat] skip ({d_lat:.3f},{d_lon:.3f}): {exc}[/yellow]")

    return saved or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathogen Scout — streaming demo.")
    parser.add_argument("--tiles", type=Path, default=config.ROOT / "data" / "sample_tiles")
    parser.add_argument("--slow", type=float, default=0.35,
                        help="Seconds to pause between tiles (set 0 to disable).")
    parser.add_argument(
        "--source",
        choices=["auto", "synthetic", "simsat"],
        default="auto",
        help=(
            "Tile source. 'simsat' = pull live tiles from the DPhi SimSat API "
            "(requires `docker compose up` from DPhi-Space/SimSat). "
            "'synthetic' = use the deterministic 8-tile demo set. "
            "'auto' (default) = try SimSat first, fall back to synthetic."
        ),
    )
    parser.add_argument("--n-simsat", type=int, default=4,
                        help="Number of live SimSat tiles to fetch when --source uses simsat.")
    args = parser.parse_args()

    if args.source in ("simsat", "auto"):
        live = _try_fetch_simsat_tiles(args.tiles, n=args.n_simsat)
        if live:
            run_demo(args.tiles, slow=args.slow)
            return
        if args.source == "simsat":
            console.print(
                "[red]SimSat is not reachable at http://localhost:9005. "
                "Start it with `docker compose up` in the DPhi-Space/SimSat repo, "
                "or rerun with --source synthetic.[/red]"
            )
            sys.exit(2)
        console.print(
            "[dim][simsat] not reachable — falling back to synthetic demo tiles.[/dim]"
        )

    run_demo(args.tiles, slow=args.slow)


if __name__ == "__main__":
    main()
