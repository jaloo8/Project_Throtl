# Session 02 -- Dashboard Improvements

**Date:** Feb 25, 2026

## Changes

**Trend arrows** -- Dashboard keeps a rolling buffer (last 30 readings). Key metrics
show `^`/`v` arrows colored green (improving) or red (degrading). Compares against
5 ticks back to filter out noise. `-` means stable.

**Health status** -- STATUS line in the header. Checks KV cache pressure, queue depth,
TTFT latency, GPU utilization, and batch utilization. Shows the specific problem
(e.g., "KV CACHE PRESSURE | QUEUE BUILDING") or "HEALTHY" if everything looks fine.

**Batch utilization** -- Changed from "8.0 / 16" to "50% (8/16)". Green >60%,
yellow 30-60%, red <30%.

## Notes

- Fixed Python 3.9 compat (`from __future__ import annotations`)
- All tests still pass
