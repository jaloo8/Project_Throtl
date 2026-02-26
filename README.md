# Throtl

A performance monitoring and optimization tool for LLM inference servers.

Built by a GPU performance engineer who got tired of seeing inference clusters run at 40% utilization.

## What it does

Throtl connects to LLM serving engines (starting with vLLM), collects performance metrics from both the serving layer and the GPU, and shows you exactly where you're wasting money -- cost per token, batch utilization, cache efficiency, latency breakdowns.

The long-term goal is closed-loop optimization: not just showing you the problem, but safely tuning your configs to fix it.

## Current status

Early development. The terminal dashboard monitors a vLLM instance (mock data for now, real integration coming) and shows:

- **Request throughput** -- running, queued, tokens/sec, batch utilization
- **Latency percentiles** -- time to first token and time per output token at p50/p95/p99
- **GPU health** -- KV cache pressure, utilization, VRAM, cost per 1K tokens
- **Trend arrows** -- each key metric shows whether it's improving or degrading
- **Health verdict** -- a single status line (HEALTHY, KV CACHE PRESSURE, QUEUE BUILDING, etc.) so you can glance and know if something needs attention

## Project structure

```
src/throtl/
  mock/          - Simulated vLLM metrics for local development
  collector/     - Metrics collection from real or mock sources
  dashboard/     - Terminal-based metrics display
tests/           - Tests
docs/chats/      - Session logs for development continuity
```

## Getting started

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run with mock data (no GPU needed)
python -m src.throtl.main --mock
```

## Requirements

- Python 3.9+
- No GPU required for development (mock mode)

## License

MIT
