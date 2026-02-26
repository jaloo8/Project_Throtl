# Session 01 -- Project Setup

**Date:** Feb 25, 2026

## Background

Starting from a research doc ("Moat-First AI Startup Ideas for a Performance Engineer")
that ranked a cross-layer LLM inference autopilot as the top opportunity. The idea:
teams running self-hosted inference waste money because they don't understand how their
serving configs interact with traffic patterns and GPU behavior. Throtl watches those
layers and eventually tunes them.

## What got built

- `InferenceSnapshot` dataclass holding everything we care about (requests, tokens/sec,
  latency percentiles, GPU cache/memory/util, cost)
- Mock generator simulating Llama 3 8B on RTX 4090 with realistic traffic patterns
- `MetricsCollector` interface so the dashboard doesn't care where data comes from
- Terminal dashboard (Rich) with three panels: requests, latency, GPU/cost
- CLI with `--mock` and `--refresh` flags
- 4 tests

## Decisions

- Collector abstraction from the start so swapping mock for real vLLM is trivial
- Mock uses seeded RNG for reproducible tests
- Rich for terminal UI since it handles live updates well
- No GPU required for any of this

## How to run

```bash
source .venv/bin/activate
python -m src.throtl.main --mock
python -m pytest tests/ -v
```
