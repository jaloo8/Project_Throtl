# Session 03 -- vLLM Scraper + SQLite Storage

**Date:** Feb 25, 2026

## What got built

**Prometheus parser** (`collector/prometheus_parser.py`) -- Parses the text format
vLLM spits out. Handles gauges, counters, and histogram buckets. Histogram percentile
estimation uses linear interpolation between bucket boundaries.

**vLLM collector** (`collector/vllm_collector.py`) -- Same interface as the mock
collector but hits a real (or fake) vLLM /metrics endpoint over HTTP. GPU util and
VRAM show as 0 since vLLM doesn't expose those -- that'll need NVML later.

**Fake vLLM server** (`mock/fake_vllm_server.py`) -- Standalone HTTP server that
responds to /metrics with Prometheus-formatted output matching real vLLM. Lets you
test the full HTTP->parse->display pipeline on a Mac.

**SQLite storage** (`storage/sqlite_store.py`) -- One row per snapshot. Saves
automatically during dashboard runs. Can disable with `--no-store`.

**CLI updates** -- `--url` flag works now, `--db` to set storage path, `--no-store`
to skip recording.

## Bugs fixed

- SQLite `datetime('now')` uses UTC but we store local time. Switched to computing
  the cutoff in Python.

## Test count

20 tests, all passing.

## How to test the full pipeline

```bash
# Terminal 1
python -m src.throtl.mock.fake_vllm_server

# Terminal 2
python -m src.throtl.main --url http://localhost:9100
```
