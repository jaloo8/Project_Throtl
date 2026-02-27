"""Tests for the config advisor."""

from datetime import datetime, timezone

from throtl.engine.config_advisor import advise
from throtl.metrics import InferenceSnapshot


def _snap(**overrides) -> InferenceSnapshot:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        requests_running=8,
        requests_waiting=0,
        tokens_per_second=80.0,
        time_to_first_token_p50=0.06,
        time_to_first_token_p95=0.12,
        time_to_first_token_p99=0.20,
        time_per_output_token_p50=0.015,
        time_per_output_token_p95=0.025,
        gpu_cache_usage_percent=0.50,
        gpu_memory_used_gb=12.0,
        gpu_memory_total_gb=24.0,
        gpu_utilization_percent=0.65,
        avg_batch_size=8.0,
        max_batch_size=16,
        estimated_cost_per_1k_tokens=0.003,
    )
    defaults.update(overrides)
    return InferenceSnapshot(**defaults)


def test_healthy_system_minimal_suggestions():
    snap = _snap()
    suggestions = advise(snap)
    # Healthy system with decent cache usage still gets prefix caching suggestion
    # but shouldn't get urgent fixes
    high_priority = [s for s in suggestions if s.priority == 1]
    assert len(high_priority) == 0


def test_high_cache_suggests_max_model_len():
    snap = _snap(gpu_cache_usage_percent=0.90)
    suggestions = advise(snap)
    flags = [s.flag for s in suggestions]
    assert "--max-model-len" in flags


def test_batch_full_with_queue():
    snap = _snap(avg_batch_size=15.0, max_batch_size=16, requests_waiting=5)
    suggestions = advise(snap)
    batch_suggestions = [s for s in suggestions if s.flag == "--max-num-seqs"]
    assert len(batch_suggestions) == 1
    assert "Increase" in batch_suggestions[0].suggested_action


def test_low_batch_suggests_decrease():
    snap = _snap(avg_batch_size=4.0, max_batch_size=16, requests_waiting=0)
    suggestions = advise(snap)
    batch_suggestions = [s for s in suggestions if s.flag == "--max-num-seqs"]
    assert len(batch_suggestions) == 1
    assert "Decrease" in batch_suggestions[0].suggested_action


def test_tight_memory_suggests_quantization():
    snap = _snap(
        gpu_memory_used_gb=22.0,
        gpu_memory_total_gb=24.0,
        gpu_cache_usage_percent=0.90,
    )
    suggestions = advise(snap)
    quant = [s for s in suggestions if "quantization" in s.flag]
    assert len(quant) == 1


def test_no_nvml_skips_memory_suggestions():
    snap = _snap(gpu_memory_used_gb=0, gpu_memory_total_gb=0)
    suggestions = advise(snap)
    mem_suggestions = [s for s in suggestions if s.flag == "--gpu-memory-utilization"]
    assert len(mem_suggestions) == 0


def test_suggestions_sorted_by_priority():
    snap = _snap(
        gpu_cache_usage_percent=0.90,
        avg_batch_size=15.0,
        max_batch_size=16,
        requests_waiting=5,
    )
    suggestions = advise(snap)
    priorities = [s.priority for s in suggestions]
    assert priorities == sorted(priorities)


def test_prefix_caching_suggested_under_load():
    snap = _snap(gpu_cache_usage_percent=0.70, requests_running=8)
    suggestions = advise(snap)
    prefix = [s for s in suggestions if "prefix" in s.flag]
    assert len(prefix) == 1
