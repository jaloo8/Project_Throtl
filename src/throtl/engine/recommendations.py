"""
Recommendation engine for vLLM tuning.

Copyright (c) 2026 JL -- see NOTICE and LICENSE files.

Looks at a recent window of snapshots and produces actionable suggestions
based on known relationships between metrics. Each rule encodes something
a GPU performance engineer would notice and suggest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from throtl.metrics import InferenceSnapshot


@dataclass
class Recommendation:
    severity: str       # "info", "warning", "critical"
    category: str       # "batching", "cache", "latency", "cost", "utilization"
    title: str
    detail: str
    config_hint: Optional[str] = None  # suggested vLLM flag or config change


def analyze(
    current: InferenceSnapshot,
    history: Optional[List[InferenceSnapshot]] = None,
) -> List[Recommendation]:
    """Run all rules against the current snapshot and recent history.

    Returns a list of recommendations sorted by severity (critical first).
    """
    recs: List[Recommendation] = []

    _check_batch_utilization(current, recs)
    _check_kv_cache_pressure(current, recs)
    _check_queue_depth(current, recs)
    _check_ttft_latency(current, recs)
    _check_tbt_latency(current, recs)
    _check_gpu_underutilization(current, recs)
    _check_cost_efficiency(current, recs)

    if history and len(history) >= 5:
        _check_latency_trend(current, history, recs)

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    recs.sort(key=lambda r: severity_order.get(r.severity, 99))
    return recs


# -- Individual rules --


def _check_batch_utilization(snap: InferenceSnapshot, recs: List[Recommendation]):
    if snap.max_batch_size == 0:
        return

    util = snap.avg_batch_size / snap.max_batch_size

    if util < 0.25 and snap.requests_waiting == 0:
        recs.append(Recommendation(
            severity="info",
            category="batching",
            title="Low batch utilization with empty queue",
            detail=(
                f"Batch utilization is {util * 100:.0f}% with no requests waiting. "
                f"The GPU has capacity for larger batches."
            ),
            config_hint="--max-num-seqs (increase to allow more concurrent sequences)",
        ))
    elif util > 0.9 and snap.requests_waiting > 0:
        recs.append(Recommendation(
            severity="warning",
            category="batching",
            title="Batch is full with requests queued",
            detail=(
                f"Running at {util * 100:.0f}% batch capacity with "
                f"{snap.requests_waiting} requests waiting. Throughput is capped."
            ),
            config_hint="--max-num-seqs (increase if VRAM allows)",
        ))


def _check_kv_cache_pressure(snap: InferenceSnapshot, recs: List[Recommendation]):
    usage = snap.gpu_cache_usage_percent

    if usage > 0.92:
        recs.append(Recommendation(
            severity="critical",
            category="cache",
            title="KV cache near full",
            detail=(
                f"KV cache at {usage * 100:.0f}%. New requests will stall or get "
                f"evicted. This directly impacts latency and throughput."
            ),
            config_hint="--gpu-memory-utilization (increase), --max-model-len (decrease), or --enable-prefix-caching",
        ))
    elif usage > 0.80:
        recs.append(Recommendation(
            severity="warning",
            category="cache",
            title="KV cache pressure building",
            detail=(
                f"KV cache at {usage * 100:.0f}%. Not critical yet, but approaching "
                f"the point where eviction starts hurting performance."
            ),
            config_hint="--enable-prefix-caching (reuse common prefixes to save cache space)",
        ))


def _check_queue_depth(snap: InferenceSnapshot, recs: List[Recommendation]):
    if snap.requests_waiting > 10:
        recs.append(Recommendation(
            severity="critical",
            category="latency",
            title="Request queue backup",
            detail=(
                f"{snap.requests_waiting} requests waiting. Users are experiencing "
                f"queuing delays. Either increase throughput or add capacity."
            ),
        ))
    elif snap.requests_waiting > 5:
        recs.append(Recommendation(
            severity="warning",
            category="latency",
            title="Queue building",
            detail=(
                f"{snap.requests_waiting} requests waiting. Queue is growing -- "
                f"watch for latency impact."
            ),
        ))


def _check_ttft_latency(snap: InferenceSnapshot, recs: List[Recommendation]):
    ttft_p95_ms = snap.time_to_first_token_p95 * 1000

    if ttft_p95_ms > 500:
        recs.append(Recommendation(
            severity="critical",
            category="latency",
            title="High time-to-first-token",
            detail=(
                f"TTFT p95 is {ttft_p95_ms:.0f}ms. Users are waiting over half a "
                f"second before seeing any output. Check queue depth and cache pressure."
            ),
        ))
    elif ttft_p95_ms > 250:
        recs.append(Recommendation(
            severity="warning",
            category="latency",
            title="Elevated time-to-first-token",
            detail=(
                f"TTFT p95 is {ttft_p95_ms:.0f}ms. Noticeable to users. Often "
                f"caused by KV cache contention or prompt processing bottleneck."
            ),
        ))


def _check_tbt_latency(snap: InferenceSnapshot, recs: List[Recommendation]):
    tbt_p95_ms = snap.time_per_output_token_p95 * 1000

    if tbt_p95_ms > 100:
        recs.append(Recommendation(
            severity="warning",
            category="latency",
            title="Slow token generation",
            detail=(
                f"Time per output token p95 is {tbt_p95_ms:.0f}ms. Streaming "
                f"responses will feel sluggish. Usually means the GPU is overloaded."
            ),
        ))


def _check_gpu_underutilization(snap: InferenceSnapshot, recs: List[Recommendation]):
    # Only flag this if we actually have GPU data (NVML connected)
    if snap.gpu_utilization_percent == 0 and snap.gpu_memory_total_gb == 0:
        return

    if snap.gpu_utilization_percent < 0.3 and snap.requests_running > 0:
        recs.append(Recommendation(
            severity="info",
            category="utilization",
            title="GPU underutilized",
            detail=(
                f"GPU at {snap.gpu_utilization_percent * 100:.0f}% utilization with "
                f"{snap.requests_running} active requests. You could handle more "
                f"concurrent load or use a smaller/cheaper GPU."
            ),
        ))


def _check_cost_efficiency(snap: InferenceSnapshot, recs: List[Recommendation]):
    if snap.tokens_per_second < 20 and snap.estimated_cost_per_1k_tokens > 0.01:
        recs.append(Recommendation(
            severity="info",
            category="cost",
            title="Low throughput driving up cost",
            detail=(
                f"Generating {snap.tokens_per_second:.0f} tokens/sec at "
                f"${snap.estimated_cost_per_1k_tokens:.4f}/1K tokens. "
                f"Improving batch utilization or reducing queue stalls would lower cost."
            ),
        ))


def _check_latency_trend(
    current: InferenceSnapshot,
    history: List[InferenceSnapshot],
    recs: List[Recommendation],
):
    """Flag if TTFT has been climbing over the recent window."""
    if len(history) < 5:
        return

    # Compare current p95 to the average of the oldest third of history
    old_slice = history[:len(history) // 3]
    old_avg_ttft = sum(s.time_to_first_token_p95 for s in old_slice) / len(old_slice)

    if old_avg_ttft == 0:
        return

    increase_pct = (current.time_to_first_token_p95 - old_avg_ttft) / old_avg_ttft

    if increase_pct > 0.5:
        recs.append(Recommendation(
            severity="warning",
            category="latency",
            title="TTFT trending upward",
            detail=(
                f"TTFT p95 has increased ~{increase_pct * 100:.0f}% over the recent "
                f"window. Likely caused by growing load or cache pressure."
            ),
        ))
