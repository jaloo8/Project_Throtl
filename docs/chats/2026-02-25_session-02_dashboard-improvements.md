# Session 02 — Dashboard Improvements

**Date:** February 25, 2026  
**Focus:** Trend arrows, health verdict, batch utilization display

---

## What changed

Three improvements to the terminal dashboard based on feedback:

### 1. Trend arrows
The dashboard now keeps a rolling buffer of the last 30 snapshots. Each key metric
(tokens/sec, queue depth, TTFT p95, TBT p95, KV cache usage, cost per 1K tokens)
shows a colored arrow indicating direction:
- Green `v` on latency/cache/cost = improving (going down)
- Green `^` on throughput = improving (going up)
- Red arrows = degrading
- Dim `-` = stable (less than 3% change)

The comparison is against the reading from 5 ticks ago (about 10 seconds at default
refresh rate), which smooths out single-tick noise.

### 2. Health verdict
A STATUS line in the header gives a quick glance assessment. It checks:
- KV cache > 92% = "KV CACHE NEAR FULL" (red)
- KV cache > 80% = "KV CACHE PRESSURE" (yellow)
- Queue > 10 = "REQUEST QUEUE BACKUP"
- Queue > 5 = "QUEUE BUILDING"
- TTFT p95 > 500ms = "HIGH TTFT LATENCY"
- TTFT p95 > 250ms = "ELEVATED TTFT"
- GPU util < 25% = "GPU UNDERUTILIZED"
- Batch util < 25% = "LOW BATCH UTILIZATION"
- No problems = "HEALTHY" (green)

Multiple problems show together, separated by pipes.

### 3. Batch utilization percentage
Instead of showing "8.0 / 16", it now shows "50% (8/16)". Color-coded:
green above 60%, yellow 30-60%, red below 30%.

## Technical notes

- Fixed Python 3.9 compatibility: used `from __future__ import annotations` and
  `Optional[X]` instead of `X | None` syntax.
- Updated README to reflect the new features.
- All 4 existing tests still pass.

## What's next

Same as before -- the real vLLM metrics scraper is the next meaningful milestone.
That's when we move from "demo" to "useful tool."
