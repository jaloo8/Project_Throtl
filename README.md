# Throtl

Performance monitoring for LLM inference servers. Connects to vLLM, collects metrics from both the serving layer and GPU, and shows you what's actually happening -- cost per token, batch utilization, cache pressure, latency percentiles.

Eventually this will do closed-loop optimization (auto-tuning configs), but right now it's focused on visibility and recommendations.

## What it shows

- Request throughput, queue depth, tokens/sec
- TTFT and TBT latency at p50/p95/p99
- KV cache usage, GPU utilization, VRAM, cost per 1K tokens
- Trend arrows showing whether each metric is improving or degrading
- Health status line (HEALTHY, KV CACHE PRESSURE, QUEUE BUILDING, etc.)
- Tuning recommendations with config hints (e.g., "increase max_num_seqs")
- Config advisor suggesting specific vLLM flags to change
- Before/after comparison to measure the impact of config changes

## Getting started

```bash
pip install -e .

# Run with fake data (no GPU needed)
throtl --mock

# Or test the full HTTP pipeline locally
python -m throtl.mock.fake_vllm_server  # terminal 1
throtl --url http://localhost:9100        # terminal 2

# One-shot config advice (no dashboard)
throtl --mock advise

# Or with Docker
docker compose up
```

## Project structure

```
src/throtl/
  mock/          - Mock data generator + fake vLLM HTTP server
  collector/     - Prometheus parser, vLLM collector, GPU monitor (NVML)
  engine/        - Recommendations, config advisor, before/after comparison
  dashboard/     - Terminal UI
  storage/       - SQLite history
tests/           - 49 tests
docs/chats/      - Dev session logs
```

## Roadmap

### Done

- [x] Terminal dashboard with live metrics, trend arrows, health status
- [x] vLLM metrics scraper (Prometheus text parser + HTTP collector)
- [x] Fake vLLM server for end-to-end testing without a GPU
- [x] SQLite storage for metrics history
- [x] NVML integration for GPU utilization, VRAM, temperature (auto-detected)
- [x] Recommendation engine with config hints
- [x] Config advisor (suggests specific vLLM flags)
- [x] Before/after comparison for measuring config change impact
- [x] Docker compose for one-command setup
- [x] CLI `advise` command for one-shot analysis

### Next up -- real hardware validation

- [ ] Test on a real GPU with vLLM serving Llama 3 8B
- [ ] Validate Prometheus parser against actual vLLM output (not just fake server)
- [ ] Tune recommendation thresholds against real workloads
- [ ] More recommendation rules as we learn from real usage

### Later -- community and multi-backend

- [ ] Support TGI or Triton as a second backend
- [ ] Time-series view of historical metrics
- [ ] OpenTelemetry integration
- [ ] Slack/webhook alerts

## Requirements

Python 3.9+. No GPU needed for development -- mock data and fake server cover the full pipeline.

## License

MIT
