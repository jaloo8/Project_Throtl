# Session 04 -- NVML, Recommendations, Config Advisor, Docker

**Date:** Feb 25, 2026

## What got built

**GPU monitor** (`collector/gpu_stats.py`) -- Reads GPU utilization, VRAM, temperature,
and power draw via NVML. Falls back gracefully on machines without a GPU.

**Recommendation engine** (`engine/recommendations.py`) -- Analyzes snapshots and
surfaces tuning suggestions. Rules cover batch utilization, KV cache pressure, queue
depth, TTFT/TBT latency, GPU underutilization, cost efficiency, and latency trends.
Each includes severity, explanation, and config hint.

**Config advisor** (`engine/config_advisor.py`) -- Suggests specific vLLM config changes
based on current metrics. Covers `--max-num-seqs`, `--max-model-len`,
`--gpu-memory-utilization`, `--enable-prefix-caching`, `--enable-chunked-prefill`,
and quantization. Includes estimated impact and confidence level.

**Before/after comparison** (`engine/comparison.py`) -- Compares two windows of
snapshots to measure the actual impact of a config change. Reports deltas, percentage
change, and whether each metric improved or regressed.

**CLI `advise` command** -- One-shot config advice without running the full dashboard:
`python -m src.throtl.main --mock advise`

**Dashboard updated** -- Recommendations panel below the metric panels. Connection
retry logic (5 attempts before giving up instead of crashing).

**Docker compose** -- `docker compose up` runs fake vLLM server + Throtl dashboard.

## Test count

49 tests, all passing.
