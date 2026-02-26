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

### Done

- [x] Terminal dashboard with live metrics, trend arrows, health status
- [x] vLLM metrics scraper (Prometheus text parser + HTTP collector)
- [x] Fake vLLM server for end-to-end testing without a GPU
- [x] SQLite storage for metrics history

### Next up -- real hardware validation

- [ ] Test on a real GPU with vLLM serving Llama 3 8B
- [ ] Add NVML integration (`pynvml`) for GPU utilization, VRAM, temperature
- [ ] Validate Prometheus parser against actual vLLM output (not just fake server)

### Then -- insight engine

The jump from "here are your numbers" to "here's what to do about them."

- [ ] First automated recommendation (e.g., "batch utilization is 69% with no queue -- increase max_num_seqs")
- [ ] KV cache pressure alerts with suggested actions
- [ ] Config diff: show what a change would do before applying it
- [ ] Before/after snapshots so users can measure impact of a change

### Later -- community and multi-backend

- [ ] Docker compose for one-command setup
- [ ] Support TGI or Triton as a second backend
- [ ] Time-series view of historical metrics
- [ ] OpenTelemetry integration
- [ ] Slack/webhook alerts

## Requirements

Python 3.9+. No GPU needed for development -- mock data and fake server cover the full pipeline.

## License

MIT
