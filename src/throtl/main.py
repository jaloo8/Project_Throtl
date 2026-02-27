"""
Throtl entry point.

Usage:
    throtl --mock                        Mock data dashboard
    throtl --url http://localhost:8000    Live vLLM dashboard
    throtl advise --mock                  One-shot config advice
"""

from __future__ import annotations

import logging

import click

from throtl import __version__
from throtl.collector.mock_collector import MockCollector
from throtl.collector.vllm_collector import VLLMCollector
from throtl.dashboard.terminal import run_dashboard, run_jsonl
from throtl.storage.sqlite_store import MetricsStore


log = logging.getLogger("throtl")


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="throtl")
@click.option("--mock", is_flag=True, default=False, help="Use simulated vLLM metrics")
@click.option("--url", default=None, help="vLLM server URL (e.g. http://localhost:8000)")
@click.option("--refresh", default=2.0, help="Dashboard refresh interval in seconds")
@click.option("--db", default="throtl_metrics.db", help="SQLite database path for history")
@click.option("--no-store", is_flag=True, default=False, help="Disable SQLite storage")
@click.option("--gpu-cost", default=1.0, help="GPU hourly rate in USD for cost tracking")
@click.option("--sla-target", default=200.0, help="TTFT SLA target in milliseconds")
@click.option("--output", type=click.Choice(["tui", "jsonl"]), default="tui",
              help="Output mode: tui (Rich dashboard) or jsonl (one JSON line per snapshot)")
@click.option("--verbose", is_flag=True, default=False, help="Enable debug logging")
@click.pass_context
def cli(ctx, mock: bool, url: str, refresh: float, db: str, no_store: bool,
        gpu_cost: float, sla_target: float, output: str, verbose: bool):
    """Throtl - LLM inference performance monitor."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ctx.ensure_object(dict)
    ctx.obj["mock"] = mock
    ctx.obj["url"] = url
    ctx.obj["refresh"] = refresh
    ctx.obj["db"] = db
    ctx.obj["no_store"] = no_store
    ctx.obj["gpu_cost"] = gpu_cost
    ctx.obj["sla_target"] = sla_target
    ctx.obj["output"] = output

    # If no subcommand, run the dashboard (backwards compatible)
    if ctx.invoked_subcommand is None:
        if not mock and not url:
            click.echo("Please specify a data source: --mock or --url <endpoint>")
            raise SystemExit(1)

        collector = VLLMCollector(base_url=url) if url else MockCollector(gpu_cost_per_hour=gpu_cost)
        store = None if no_store else MetricsStore(db_path=db)

        runner = run_jsonl if output == "jsonl" else run_dashboard

        try:
            runner(collector, refresh_interval=refresh, store=store)
        finally:
            if store:
                store.close()
            if hasattr(collector, "close"):
                collector.close()


@cli.command()
@click.pass_context
def advise(ctx):
    """Take a single snapshot and print config recommendations."""
    from rich.console import Console
    from rich.table import Table
    from throtl.engine.config_advisor import advise as get_advice
    from throtl.engine.recommendations import analyze

    mock = ctx.obj["mock"]
    url = ctx.obj["url"]
    gpu_cost = ctx.obj["gpu_cost"]

    if not mock and not url:
        click.echo("Please specify a data source: --mock or --url <endpoint>")
        raise SystemExit(1)

    collector = VLLMCollector(base_url=url) if url else MockCollector(gpu_cost_per_hour=gpu_cost)

    try:
        snapshot = collector.collect()
    finally:
        if hasattr(collector, "close"):
            collector.close()

    console = Console()

    # Health recommendations
    recs = analyze(snapshot)
    if recs:
        console.print("\n[bold]Current issues:[/bold]")
        for rec in recs:
            color = {"critical": "red", "warning": "yellow", "info": "dim"}.get(rec.severity, "white")
            console.print(f"  [{color}]{rec.severity.upper()}[/{color}]  {rec.title}")
            console.print(f"           [dim]{rec.detail}[/dim]")
    else:
        console.print("\n[bold green]No issues detected.[/bold green]")

    # Config suggestions
    suggestions = get_advice(snapshot)
    if suggestions:
        console.print("\n[bold]Config suggestions:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Priority", width=8, justify="center")
        table.add_column("Flag")
        table.add_column("Action")
        table.add_column("Expected impact")
        table.add_column("Confidence", width=10)

        for s in suggestions:
            table.add_row(
                str(s.priority),
                f"[cyan]{s.flag}[/cyan]",
                s.suggested_action,
                s.expected_impact,
                s.confidence,
            )
        console.print(table)
    else:
        console.print("\n[dim]No config changes suggested -- looks well-tuned.[/dim]")

    console.print()


if __name__ == "__main__":
    cli()
