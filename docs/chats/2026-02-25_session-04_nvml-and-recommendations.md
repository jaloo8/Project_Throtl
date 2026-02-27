# Session 04 -- NVML Integration + Recommendation Engine

**Date:** Feb 25, 2026

## What got built

**GPU monitor** (`collector/gpu_stats.py`) -- Reads GPU utilization, VRAM, temperature,
and power draw via NVML (the library behind nvidia-smi). Falls back gracefully on
machines without a GPU -- returns None instead of crashing.

**VLLMCollector updated** -- Now uses the GPU monitor to fill in utilization and memory
fields that vLLM's /metrics endpoint doesn't expose. On Mac (no GPU), these stay at
zero like before. On a machine with a GPU, they populate automatically.

**Recommendation engine** (`engine/recommendations.py`) -- Analyzes snapshots and
produces actionable tuning suggestions. Current rules:

- Batch utilization (low with empty queue, or full with requests queued)
- KV cache pressure (warning at 80%, critical at 92%)
- Queue depth (warning >5, critical >10)
- TTFT latency (warning >250ms p95, critical >500ms)
- TBT latency (warning >100ms p95)
- GPU underutilization (only when NVML data is present)
- Cost efficiency (low throughput driving up cost)
- TTFT trend (flags if p95 is climbing over the recent window)

Each recommendation includes severity, category, a human-readable explanation, and
an optional config hint (e.g., which vLLM flag to change).

**Dashboard updated** -- Recommendations panel shows below the three metric panels.
Sorted by severity, capped at 6 items, color-coded. Green border when healthy.

## Test count

35 tests, all passing. New tests cover GPU monitor fallback (3) and all
recommendation rules (12).
