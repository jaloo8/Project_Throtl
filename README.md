# Throtl

Performance monitoring for LLM inference servers. Connects to vLLM, collects metrics from both the serving layer and GPU, and shows you what's actually happening -- cost per token, batch utilization, cache pressure, latency percentiles.

Eventually this will do closed-loop optimization (auto-tuning configs), but right now it's focused on visibility.

## What it shows

- Request throughput, queue depth, tokens/sec
- TTFT and TBT latency at p50/p95/p99
- KV cache usage, GPU utilization, VRAM, cost per 1K tokens
- Trend arrows showing whether each metric is improving or degrading
- Health status line (HEALTHY, KV CACHE PRESSURE, QUEUE BUILDING, etc.)

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run with fake data (no GPU needed)
python -m src.throtl.main --mock

# Or test the full HTTP pipeline locally
python -m src.throtl.mock.fake_vllm_server  # terminal 1
python -m src.throtl.main --url http://localhost:9100  # terminal 2
```

## Project structure

```
src/throtl/
  mock/          - Mock data generator + fake vLLM HTTP server
  collector/     - Prometheus parser, vLLM collector, mock collector
  dashboard/     - Terminal UI
  storage/       - SQLite history
tests/           - 20 tests
docs/chats/      - Dev session logs
```

## Roadmap

- [x] Terminal dashboard with live metrics
- [x] Health detection + trend arrows
- [x] vLLM metrics scraper (Prometheus parser)
- [x] SQLite history storage
- [ ] Time-series view
- [ ] Automated tuning recommendations
- [ ] OpenTelemetry integration
- [ ] Multi-backend support (Triton, TensorRT-LLM)

## Requirements

Python 3.9+. No GPU needed for development.

## License

MIT
