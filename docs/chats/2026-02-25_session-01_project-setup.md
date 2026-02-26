# Session 01 — Project Setup

**Date:** February 25, 2026  
**Focus:** Initial project scaffolding, mock metrics generator, terminal dashboard

---

## Context

This is the first working session for Throtl. The idea comes from a research document
("Moat-First AI Startup Ideas for a Performance Engineer") that identified a
"Cross-Layer LLM Inference Autopilot" as the top opportunity for a GPU performance
engineer to build a side project into a real product.

The core thesis: most teams running self-hosted LLM inference waste money because they
don't understand the interaction between their serving configs, traffic patterns, and
GPU utilization. Throtl monitors those layers and (eventually) tunes them automatically.

## What we discussed before coding

- Reviewed the full 22-page research PDF and gave an honest multi-perspective assessment
  (VC, CEO, CFO angles).
- Key strengths: real problem, strong technical fit for a GPU perf engineer, data flywheel
  moat is structurally sound.
- Key concerns: data flywheel cold-start problem, competitive window may be narrow as
  serving frameworks add native optimization, enterprise sales is hard for a solo founder,
  10-12 week MVP estimate is aggressive for a side project.
- Agreed to start with the smallest useful thing: a metrics collector and dashboard for
  vLLM, using mock data so development can happen on an M1 Mac.

## What we built

### Project structure
```
Project_Throtl/
  src/throtl/
    metrics.py           - Core metric definitions (InferenceSnapshot dataclass)
    main.py              - CLI entry point (click-based)
    mock/
      generator.py       - Simulated vLLM server producing realistic metrics
    collector/
      base.py            - Abstract collector interface
      mock_collector.py  - Collector wrapping the mock generator
    dashboard/
      terminal.py        - Rich-based live terminal dashboard
  tests/
    test_mock_generator.py  - 4 tests covering snapshot validity, accumulation,
                              determinism, and summary format
  docs/chats/             - Session logs (you're reading one)
  requirements.txt        - rich, httpx, click
  README.md               - Updated with project description and quickstart
```

### Key design decisions

1. **Collector abstraction** — `MetricsCollector` is an abstract base class. The mock
   and (future) real vLLM collector both implement the same interface. This means the
   dashboard doesn't care where data comes from.

2. **Mock generator models a Llama 3 8B on RTX 4090** — Sinusoidal traffic patterns
   with random spikes, queue buildup under load, KV cache pressure correlated with
   active sequences, latency degradation under high utilization. Seeded RNG for
   reproducible tests.

3. **Terminal dashboard using Rich** — Three-panel layout showing requests, latency
   (p50/p95/p99), and GPU stats. Color-coded thresholds (green/yellow/red). Refreshes
   every 2 seconds by default.

4. **No GPU required for development** — Everything runs on the M1 Mac using mock data.
   Real vLLM connection is stubbed out in the CLI but not implemented yet.

### Dependencies
- `rich` — terminal UI rendering
- `httpx` — will be used for real vLLM metrics endpoint (not used yet)
- `click` — CLI argument parsing
- `pytest` — testing (dev dependency)

## How to run

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the dashboard with mock data
python -m src.throtl.main --mock

# Run tests
python -m pytest tests/ -v
```

## What's next

- **Real vLLM collector** — Connect to a live vLLM /metrics endpoint and parse
  Prometheus-format output into InferenceSnapshot.
- **Metrics history** — Store snapshots over time (SQLite to start) so we can show
  trends, not just current state.
- **Alerts/flags** — Simple rules like "cache usage > 90% for 5 minutes" or
  "queue depth growing steadily."
- **When to use the Windows/NVIDIA laptop** — Once we have the real collector ready,
  test against an actual vLLM instance running in WSL2.

## Notes

- All development is on an M1 Pro MacBook for now.
- Security: no packages installed globally, no remote scripts executed, no ports exposed,
  no credentials stored.
- The `.venv/` directory is gitignored.
