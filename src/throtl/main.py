"""
Throtl entry point.

Usage:
    python -m src.throtl.main --mock          Run with simulated metrics
    python -m src.throtl.main --url <url>     Connect to a real vLLM server (coming soon)
"""

import click

from src.throtl.collector.mock_collector import MockCollector
from src.throtl.dashboard.terminal import run_dashboard


@click.command()
@click.option("--mock", is_flag=True, default=False, help="Use simulated vLLM metrics")
@click.option("--url", default=None, help="vLLM metrics endpoint URL (e.g. http://localhost:8000/metrics)")
@click.option("--refresh", default=2.0, help="Dashboard refresh interval in seconds")
def main(mock: bool, url: str, refresh: float):
    """Throtl - LLM inference performance monitor."""

    if url:
        click.echo("Live vLLM connection is not implemented yet.")
        click.echo("Use --mock to run with simulated data for now.")
        raise SystemExit(1)

    if not mock and not url:
        click.echo("Please specify a data source: --mock or --url <endpoint>")
        raise SystemExit(1)

    collector = MockCollector()
    run_dashboard(collector, refresh_interval=refresh)


if __name__ == "__main__":
    main()
