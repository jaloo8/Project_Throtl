"""
Throtl entry point.

Usage:
    python -m src.throtl.main --mock                        Mock data dashboard
    python -m src.throtl.main --url http://localhost:8000    Live vLLM dashboard
    python -m src.throtl.mock.fake_vllm_server               Start fake vLLM server
"""

from __future__ import annotations

import click

from src.throtl.collector.mock_collector import MockCollector
from src.throtl.collector.vllm_collector import VLLMCollector
from src.throtl.dashboard.terminal import run_dashboard
from src.throtl.storage.sqlite_store import MetricsStore


@click.command()
@click.option("--mock", is_flag=True, default=False, help="Use simulated vLLM metrics")
@click.option("--url", default=None, help="vLLM server URL (e.g. http://localhost:8000)")
@click.option("--refresh", default=2.0, help="Dashboard refresh interval in seconds")
@click.option("--db", default="throtl_metrics.db", help="SQLite database path for history")
@click.option("--no-store", is_flag=True, default=False, help="Disable SQLite storage")
def main(mock: bool, url: str, refresh: float, db: str, no_store: bool):
    """Throtl - LLM inference performance monitor."""

    if not mock and not url:
        click.echo("Please specify a data source: --mock or --url <endpoint>")
        raise SystemExit(1)

    if url:
        collector = VLLMCollector(base_url=url)
    else:
        collector = MockCollector()

    store = None
    if not no_store:
        store = MetricsStore(db_path=db)

    try:
        run_dashboard(collector, refresh_interval=refresh, store=store)
    finally:
        if store:
            store.close()
        if hasattr(collector, "close"):
            collector.close()


if __name__ == "__main__":
    main()
