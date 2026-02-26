"""
Terminal dashboard for Throtl.

Displays a live-updating view of inference metrics using Rich.
Nothing fancy -- just the numbers that matter, refreshed every 2 seconds.
"""

import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from src.throtl.collector.base import MetricsCollector
from src.throtl.metrics import InferenceSnapshot


def _color_for_percent(value: float) -> str:
    """Green when low, yellow when moderate, red when high."""
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


def build_display(snapshot: InferenceSnapshot, source_name: str) -> Layout:
    """Build the full terminal display from a single snapshot."""

    layout = Layout()

    # Header
    header = Text(f"  throtl  |  {source_name}", style="bold white on blue")
    header.append(f"\n  {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}", style="dim")

    # Request stats
    req_table = Table(show_header=True, header_style="bold cyan", expand=True)
    req_table.add_column("Metric", style="dim")
    req_table.add_column("Value", justify="right")
    req_table.add_row("Running", str(snapshot.requests_running))
    req_table.add_row("Waiting", f"[{'red' if snapshot.requests_waiting > 5 else 'green'}]{snapshot.requests_waiting}[/]")
    req_table.add_row("Completed (total)", f"{snapshot.requests_completed:,}")
    req_table.add_row("Tokens/sec", f"[bold]{snapshot.tokens_per_second:.1f}[/bold]")
    req_table.add_row("Avg batch size", f"{snapshot.avg_batch_size:.1f} / {snapshot.max_batch_size}")

    # Latency stats
    lat_table = Table(show_header=True, header_style="bold cyan", expand=True)
    lat_table.add_column("Latency", style="dim")
    lat_table.add_column("p50", justify="right")
    lat_table.add_column("p95", justify="right")
    lat_table.add_column("p99", justify="right")

    ttft_p50_ms = snapshot.time_to_first_token_p50 * 1000
    ttft_p95_ms = snapshot.time_to_first_token_p95 * 1000
    ttft_p99_ms = snapshot.time_to_first_token_p99 * 1000
    tbt_p50_ms = snapshot.time_per_output_token_p50 * 1000
    tbt_p95_ms = snapshot.time_per_output_token_p95 * 1000
    tbt_p99_ms = snapshot.time_per_output_token_p99 * 1000

    lat_table.add_row(
        "Time to first token",
        f"[{_color_for_latency_ms(ttft_p50_ms)}]{ttft_p50_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(ttft_p95_ms)}]{ttft_p95_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(ttft_p99_ms)}]{ttft_p99_ms:.1f}ms[/]",
    )
    lat_table.add_row(
        "Time per output token",
        f"[{_color_for_latency_ms(tbt_p50_ms, 50)}]{tbt_p50_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(tbt_p95_ms, 50)}]{tbt_p95_ms:.1f}ms[/]",
        f"[{_color_for_latency_ms(tbt_p99_ms, 50)}]{tbt_p99_ms:.1f}ms[/]",
    )

    # GPU stats
    gpu_table = Table(show_header=True, header_style="bold cyan", expand=True)
    gpu_table.add_column("GPU", style="dim")
    gpu_table.add_column("Value", justify="right")

    cache_color = _color_for_percent(snapshot.gpu_cache_usage_percent)
    util_color = _color_for_percent(snapshot.gpu_utilization_percent)

    gpu_table.add_row("KV Cache usage", f"[{cache_color}]{snapshot.gpu_cache_usage_percent * 100:.1f}%[/]")
    gpu_table.add_row("GPU utilization", f"[{util_color}]{snapshot.gpu_utilization_percent * 100:.1f}%[/]")
    gpu_table.add_row("VRAM", f"{snapshot.gpu_memory_used_gb:.1f} / {snapshot.gpu_memory_total_gb:.1f} GB")
    gpu_table.add_row("Cost per 1K tokens", f"${snapshot.estimated_cost_per_1k_tokens:.4f}")

    # Assemble layout
    layout.split_column(
        Layout(Panel(header, border_style="blue"), size=4),
        Layout(name="body"),
        Layout(Panel(Text("  Press Ctrl+C to stop", style="dim"), border_style="dim"), size=3),
    )

    layout["body"].split_row(
        Layout(Panel(req_table, title="Requests", border_style="cyan")),
        Layout(Panel(lat_table, title="Latency", border_style="cyan")),
        Layout(Panel(gpu_table, title="GPU & Cost", border_style="cyan")),
    )

    return layout


def run_dashboard(collector: MetricsCollector, refresh_interval: float = 2.0):
    """Run the live terminal dashboard. Blocks until Ctrl+C."""

    console = Console()
    source_name = collector.name()

    console.print(f"\n[bold]Starting Throtl dashboard...[/bold]")
    console.print(f"Source: {source_name}")
    console.print(f"Refresh: every {refresh_interval}s\n")
    time.sleep(1)

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        try:
            while True:
                snapshot = collector.collect()
                display = build_display(snapshot, source_name)
                live.update(display)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            pass

    console.print("\n[dim]Dashboard stopped.[/dim]")
