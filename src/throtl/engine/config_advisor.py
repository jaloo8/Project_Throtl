"""
Config advisor for vLLM tuning.

Copyright (c) 2026 JL -- see NOTICE and LICENSE files.

Takes the current snapshot state and produces a ranked list of config
changes worth trying, along with estimated impact. These are heuristics
based on known vLLM behavior -- not guarantees. The before/after
comparison module is what actually measures whether a change helped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from throtl.metrics import InferenceSnapshot


@dataclass
class ConfigSuggestion:
    flag: str              # vLLM CLI flag or env var
    current_hint: str      # what the current state looks like
    suggested_action: str  # what to change
    expected_impact: str   # plain English prediction
    confidence: str        # "high", "medium", "low"
    priority: int          # 1 = try first, higher = less urgent


def advise(snapshot: InferenceSnapshot) -> List[ConfigSuggestion]:
    """Analyze current metrics and suggest config changes worth trying.

    Returns suggestions sorted by priority (most impactful first).
    """
    suggestions: List[ConfigSuggestion] = []

    _advise_batch_size(snapshot, suggestions)
    _advise_cache_config(snapshot, suggestions)
    _advise_gpu_memory(snapshot, suggestions)
    _advise_quantization(snapshot, suggestions)
    _advise_prefix_caching(snapshot, suggestions)

    suggestions.sort(key=lambda s: s.priority)
    return suggestions


def _advise_batch_size(snap: InferenceSnapshot, out: List[ConfigSuggestion]):
    if snap.max_batch_size == 0:
        return

    util = snap.avg_batch_size / snap.max_batch_size

    if util < 0.4 and snap.requests_waiting == 0:
        out.append(ConfigSuggestion(
            flag="--max-num-seqs",
            current_hint=f"Batch {util * 100:.0f}% utilized, no queue",
            suggested_action="Decrease to match actual demand (saves memory for KV cache)",
            expected_impact="Frees VRAM for longer sequences or larger cache",
            confidence="high",
            priority=3,
        ))

    elif util > 0.85 and snap.requests_waiting > 3:
        out.append(ConfigSuggestion(
            flag="--max-num-seqs",
            current_hint=f"Batch {util * 100:.0f}% full, {snap.requests_waiting} queued",
            suggested_action="Increase if VRAM allows (watch gpu_cache_usage)",
            expected_impact="Reduces queue wait time, improves TTFT",
            confidence="medium",
            priority=1,
        ))


def _advise_cache_config(snap: InferenceSnapshot, out: List[ConfigSuggestion]):
    cache = snap.gpu_cache_usage_percent

    if cache > 0.85:
        out.append(ConfigSuggestion(
            flag="--max-model-len",
            current_hint=f"KV cache at {cache * 100:.0f}%",
            suggested_action="Reduce max sequence length if your workload allows shorter contexts",
            expected_impact="Lowers peak cache usage, prevents eviction stalls",
            confidence="medium",
            priority=1,
        ))

    if cache > 0.70 and snap.time_to_first_token_p95 > 0.15:
        out.append(ConfigSuggestion(
            flag="--enable-chunked-prefill",
            current_hint=f"Cache at {cache * 100:.0f}%, TTFT p95 at {snap.time_to_first_token_p95 * 1000:.0f}ms",
            suggested_action="Enable chunked prefill to overlap prompt processing with generation",
            expected_impact="Smooths out TTFT spikes under high cache pressure",
            confidence="medium",
            priority=2,
        ))


def _advise_gpu_memory(snap: InferenceSnapshot, out: List[ConfigSuggestion]):
    if snap.gpu_memory_total_gb == 0:
        return  # no NVML data

    mem_usage = snap.gpu_memory_used_gb / snap.gpu_memory_total_gb

    if mem_usage < 0.6 and snap.gpu_cache_usage_percent > 0.80:
        out.append(ConfigSuggestion(
            flag="--gpu-memory-utilization",
            current_hint=f"VRAM {mem_usage * 100:.0f}% used but cache at {snap.gpu_cache_usage_percent * 100:.0f}%",
            suggested_action="Increase from default 0.9 to 0.95 to give more space to KV cache",
            expected_impact="More cache headroom, fewer evictions",
            confidence="high",
            priority=1,
        ))


def _advise_quantization(snap: InferenceSnapshot, out: List[ConfigSuggestion]):
    if snap.gpu_memory_total_gb == 0:
        return

    mem_usage = snap.gpu_memory_used_gb / snap.gpu_memory_total_gb

    # If memory is very tight and cache is under pressure, quantization might help
    if mem_usage > 0.9 and snap.gpu_cache_usage_percent > 0.85:
        out.append(ConfigSuggestion(
            flag="--quantization awq / --quantization gptq",
            current_hint=f"VRAM at {mem_usage * 100:.0f}%, cache at {snap.gpu_cache_usage_percent * 100:.0f}%",
            suggested_action="Use a quantized model variant to free VRAM for KV cache",
            expected_impact="~50% model memory reduction, minor quality tradeoff",
            confidence="low",
            priority=4,
        ))


def _advise_prefix_caching(snap: InferenceSnapshot, out: List[ConfigSuggestion]):
    # Useful when cache is busy and there's likely shared system prompts
    if snap.gpu_cache_usage_percent > 0.60 and snap.requests_running > 4:
        out.append(ConfigSuggestion(
            flag="--enable-prefix-caching",
            current_hint=f"Cache at {snap.gpu_cache_usage_percent * 100:.0f}% with {snap.requests_running} active",
            suggested_action="Enable automatic prefix caching for shared prompt prefixes",
            expected_impact="Reduces cache duplication if requests share common system prompts",
            confidence="medium",
            priority=2,
        ))
