# Throtl

A performance monitoring and optimization tool for LLM inference servers.

Built by a GPU performance engineer who got tired of seeing inference clusters run at 40% utilization.

## What it does

Throtl connects to LLM serving engines (starting with vLLM), collects performance metrics from both the serving layer and the GPU, and shows you exactly where you're wasting money -- cost per token, batch utilization, cache efficiency, latency breakdowns.

The long-term goal is closed-loop optimization: not just showing you the problem, but safely tuning your configs to fix it.

## Current status

Early development. We're building the metrics collection and visualization layer first.

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

- Python 3.10+
- No GPU required for development (mock mode)

## License

MIT
