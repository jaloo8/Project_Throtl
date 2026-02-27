"""Terminal dashboard using Rich. Shows metrics, trend arrows, and health status."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from throtl import __version__
from throtl.collector.base import MetricsCollector
from throtl.engine.recommendations import Recommendation, analyze
from throtl.metrics import InferenceSnapshot
from throtl.storage.sqlite_store import MetricsStore


# How many snapshots to keep for trend comparison
HISTORY_SIZE = 30


def _color_for_percent(value: float) -> str:
    if value < 0.5:
        return "green"
    elif value < 0.8:
        return "yellow"
    return "red"


def _color_for_latency_ms(value_ms: float, threshold_ms: float = 200) -> str:
    if value_ms < threshold_ms * 0.5:
        return "green"
    elif value_ms < threshold_ms:
        return "yellow"
    return "red"


def _trend_arrow(current: float, previous: float, better_when: str = "lower") -> str:
    """Returns a colored ^ or v arrow. Green = improving, red = degrading."""
    if previous == 0:
        return ""

    pct_change = (current - previous) / abs(previous)
    threshold = 0.03  # ignore noise below 3%

    if abs(pct_change) < threshold:
        return "[dim]-[/dim]"

    going_up = pct_change > 0

    if better_when == "lower":
        color = "red" if going_up else "green"
    else:
        color = "green" if going_up else "red"

    arrow = "^" if going_up else "v"
    return f"[{color}]{arrow}[/{color}]"


def _evaluate_health(snapshot: InferenceSnapshot) -> tuple[str, str]:
    """Check key signals and return (status_text, rich_style) for the header."""
    problems = []

    if snapshot.gpu_cache_usage_percent > 0.92:
        problems.append("KV CACHE NEAR FULL")
    elif snapshot.gpu_cache_usage_percent > 0.80:
        problems.append("KV CACHE PRESSURE")

    if snapshot.requests_waiting > 10:
        problems.append("REQUEST QUEUE BACKUP")
    elif snapshot.requests_waiting > 5:
        problems.append("QUEUE BUILDING")

    ttft_p95_ms = snapshot.time_to_first_token_p95 * 1000
    if ttft_p95_ms > 500:
        problems.append("HIGH TTFT LATENCY")
    elif ttft_p95_ms > 250:
        problems.append("ELEVATED TTFT")

    if snapshot.gpu_utilization_percent < 0.25:
        problems.append("GPU UNDERUTILIZED")

    batch_util = snapshot.avg_batch_size / max(1, snapshot.max_batch_size)
    if batch_util < 0.25:
        problems.append("LOW BATCH UTILIZATION")

    if not problems:
        return "HEALTHY", "bold green"

    severity = "bold yellow" if len(problems) == 1 else "bold red"
    return " | ".join(problems), severity


def _get_lookback(history: deque, steps_back: int = 5) -> Optional[InferenceSnapshot]:
    """Grab a snapshot from N steps ago for trend comparison."""
    if len(history) > steps_back:
        return history[-(steps_back + 1)]
    elif len(history) > 1:
        return history[0]
    return None


def build_display(
    snapshot: InferenceSnapshot,
    source_name: str,
    history: deque,
) -> Layout:

    layout = Layout()
    prev = _get_lookback(history)

    status_text, status_style = _evaluate_health(snapshot)
    header = Text(f"  throtl v{__version__}  |  {source_name}", style="bold white on blue")
    header.append(f"\n  {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  ", style="dim")
    header.append(f"  STATUS: {status_text}", style=status_style)

    # -- Requests panel --
    req_table = Table(show_header=True, header_style="bold cyan", expand=True)
    req_table.add_column("Metric", style="dim")
    req_table.add_column("Value", justify="right")
    req_table.add_column("", width=2)

    tps_trend = _trend_arrow(snapshot.tokens_per_second, prev.tokens_per_second, "higher") if prev else ""
    wait_trend = _trend_arrow(snapshot.requests_waiting, prev.requests_waiting, "lower") if prev else ""

    req_table.add_row("Running", str(snapshot.requests_running), "")
    req_table.add_row(
        "Waiting",
        f"[{'red' if snapshot.requests_waiting > 5 else 'green'}]{snapshot.requests_waiting}[/]",
        wait_trend,
    )
    req_table.add_row("Completed (total)", f"{snapshot.requests_completed:,}", "")
    req_table.add_row("Tokens/sec", f"[bold]{snapshot.tokens_per_second:.1f}[/bold]", tps_trend)

    batch_util = snapshot.avg_batch_size / max(1, snapshot.max_batch_size)
    batch_color = "green" if batch_util > 0.6 else ("yellow" if batch_util > 0.3 else "red")
    req_table.add_row(
        "Batch utilization",
        f"[{batch_color}]{batch_util * 100:.0f}%[/{batch_color}]  ({snapshot.avg_batch_size:.0f}/{snapshot.max_batch_size})",
        "",
    )

    # -- Latency panel --
    lat_table = Table(show_header=True, header_style="bold cyan", expand=True)
    lat_table.add_column("Latency", style="dim")
    lat_table.add_column("p50", justify="right")
    lat_table.add_column("p95", justify="right")
    lat_table.add_column("p99", justify="right")
    lat_table.add_column("", width=2)

    ttft_p50_ms = snapshot.time_to_first_token_p50 * 1000
    ttft_p95_ms = snapshot.time_to_first_token_p95 * 1000
    ttft_p99_ms = snapshot.time_to_first_token_p99 * 1000
    tbt_p50_ms = snapshot.time_per_output_token_p50 * 1000
    tbt_p95_ms = snapshot.time_per_output_token_p95 * 1000
    tbt_p99_ms = snapshot.time_per_output_token_p99 * 1000

    ttft_trend = _trend_arrow(ttft_p95_ms, prev.time_to_first_token_p95 * 1000, "lower") if prev else ""
    tbt_trend = _trend_arrow(tbt_p95_ms, prev.time_per_output_token_p95 * 1000, "lower") if prev else ""

    lat_table.add_row(
        "Time to first token",
        f"[{_color_for_latency_ms(ttft_p50_ms)}]{ttft_p50_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(ttft_p95_ms)}]{ttft_p95_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(ttft_p99_ms)}]{ttft_p99_ms:.1f}ms[/]",
        ttft_trend,
    )
    lat_table.add_row(
        "Time per output token",
        f"[{_color_for_latency_ms(tbt_p50_ms, 50)}]{tbt_p50_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(tbt_p95_ms, 50)}]{tbt_p95_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(tbt_p99_ms, 50)}]{tbt_p99_ms:.1f}ms[/]",
        tbt_trend,
    )

    # -- GPU & cost panel --
    gpu_table = Table(show_header=True, header_style="bold cyan", expand=True)
    gpu_table.add_column("GPU", style="dim")
    gpu_table.add_column("Value", justify="right")
    gpu_table.add_column("", width=2)

    cache_color = _color_for_percent(snapshot.gpu_cache_usage_percent)
    util_color = _color_for_percent(snapshot.gpu_utilization_percent)

    cache_trend = _trend_arrow(snapshot.gpu_cache_usage_percent, prev.gpu_cache_usage_percent, "lower") if prev else ""
    cost_trend = _trend_arrow(snapshot.estimated_cost_per_1k_tokens, prev.estimated_cost_per_1k_tokens, "lower") if prev else ""

    gpu_table.add_row(
        "KV Cache usage",
        f"[{cache_color}]{snapshot.gpu_cache_usage_percent * 100:.1f}%[/]",
        cache_trend,
    )
    gpu_table.add_row(
        "GPU utilization",
        f"[{util_color}]{snapshot.gpu_utilization_percent * 100:.1f}%[/]",
        "",
    )
    gpu_table.add_row(
        "VRAM",
        f"{snapshot.gpu_memory_used_gb:.1f} / {snapshot.gpu_memory_total_gb:.1f} GB",
        "",
    )
    gpu_table.add_row(
        "Cost per 1K tokens",
        f"${snapshot.estimated_cost_per_1k_tokens:.4f}",
        cost_trend,
    )

    # -- Recommendations panel --
    recs = analyze(current=snapshot, history=list(history))
    rec_panel = _build_rec_panel(recs)

    layout.split_column(
        Layout(Panel(header, border_style="blue"), size=4),
        Layout(name="body"),
        Layout(rec_panel, size=max(5, min(len(recs) + 3, 10))),
        Layout(Panel(Text("  Press Ctrl+C to stop", style="dim"), border_style="dim"), size=3),
    )

    layout["body"].split_row(
        Layout(Panel(req_table, title="Requests", border_style="cyan")),
        Layout(Panel(lat_table, title="Latency", border_style="cyan")),
        Layout(Panel(gpu_table, title="GPU & Cost", border_style="cyan")),
    )

    return layout


_SEVERITY_STYLE = {
    "critical": "bold red",
    "warning": "yellow",
    "info": "dim",
}


def _build_rec_panel(recs: list[Recommendation]) -> Panel:
    """Build a panel showing current tuning recommendations."""
    if not recs:
        content = Text("  No issues detected", style="bold green")
        return Panel(content, title="Recommendations", border_style="green")

    table = Table(show_header=False, expand=True, padding=(0, 1))
    table.add_column("sev", width=8)
    table.add_column("recommendation")

    for rec in recs[:6]:  # cap at 6 to keep the panel compact
        sev_style = _SEVERITY_STYLE.get(rec.severity, "dim")
        label = f"[{sev_style}]{rec.severity.upper()}[/{sev_style}]"
        detail = rec.title
        if rec.config_hint:
            detail += f"  [dim]({rec.config_hint})[/dim]"
        table.add_row(label, detail)

    border = "red" if recs[0].severity == "critical" else "yellow"
    return Panel(table, title=f"Recommendations ({len(recs)})", border_style=border)


def run_dashboard(
    collector: MetricsCollector,
    refresh_interval: float = 2.0,
    store: Optional[MetricsStore] = None,
):

    console = Console()
    source_name = collector.name()
    history: deque[InferenceSnapshot] = deque(maxlen=HISTORY_SIZE)

    log.info("Starting dashboard: source=%s, refresh=%.1fs", source_name, refresh_interval)
    console.print(f"\n[bold]Starting Throtl v{__version__}...[/bold]")
    console.print(f"Source: {source_name}")
    console.print(f"Refresh: every {refresh_interval}s")
    if store:
        console.print(f"Storage: SQLite (recording history)")
    console.print()
    time.sleep(1)

    consecutive_errors = 0

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                try:
                    snapshot = collector.collect()
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    log.warning("Collection failed (attempt %d/5): %s", consecutive_errors, e)
                    if consecutive_errors >= 5:
                        log.error("Lost connection after 5 retries, exiting")
                        console.print(f"\n[bold red]Lost connection after 5 retries: {e}[/bold red]")
                        break
                    # Show error in dashboard but keep trying
                    error_text = Text(f"  Connection error (retry {consecutive_errors}/5): {e}", style="bold red")
                    live.update(Panel(error_text, border_style="red"))
                    time.sleep(refresh_interval)
                    continue

                history.append(snapshot)
                if store:
                    store.save(snapshot)
                display = build_display(snapshot, source_name, history)
                live.update(display)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            pass

    if store:
        count = store.count()
        console.print(f"\n[dim]Dashboard stopped. {count} snapshots recorded.[/dim]")
    else:
        console.print("\n[dim]Dashboard stopped.[/dim]")


def run_jsonl(
    collector: MetricsCollector,
    refresh_interval: float = 2.0,
    store: Optional[MetricsStore] = None,
):
    """Non-interactive output mode: prints one JSON object per snapshot per line.

    Designed for Docker, CI pipelines, and log aggregators where a Rich TUI
    isn't available.
    """
    import json
    import sys

    source_name = collector.name()
    log.info("Starting JSONL output: source=%s, refresh=%.1fs", source_name, refresh_interval)

    consecutive_errors = 0

    try:
        while True:
            try:
                snapshot = collector.collect()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                log.warning("Collection failed (attempt %d/5): %s", consecutive_errors, e)
                if consecutive_errors >= 5:
                    log.error("Lost connection after 5 retries, exiting")
                    break
                time.sleep(refresh_interval)
                continue

            if store:
                store.save(snapshot)

            record = snapshot.summary()
            record["source"] = source_name
            sys.stdout.write(json.dumps(record) + "\n")
            sys.stdout.flush()
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        pass
