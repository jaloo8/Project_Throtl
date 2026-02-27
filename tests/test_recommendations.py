"""Tests for the recommendation engine."""

from datetime import datetime

from src.throtl.engine.recommendations import Recommendation, analyze
from src.throtl.metrics import InferenceSnapshot


def _make_snapshot(**overrides) -> InferenceSnapshot:
    defaults = dict(
        timestamp=datetime.now(),
        requests_running=8,
        requests_waiting=0,
        requests_completed=100,
        prompt_tokens_total=5000,
        generation_tokens_total=3000,
        tokens_per_second=80.0,
        time_to_first_token_p50=0.06,
        time_to_first_token_p95=0.12,
        time_to_first_token_p99=0.20,
        time_per_output_token_p50=0.015,
        time_per_output_token_p95=0.025,
        time_per_output_token_p99=0.035,
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


def test_healthy_snapshot_no_recommendations():
    snap = _make_snapshot()
    recs = analyze(snap)
    assert len(recs) == 0


def test_kv_cache_critical():
    snap = _make_snapshot(gpu_cache_usage_percent=0.95)
    recs = analyze(snap)
    categories = [r.category for r in recs]
    assert "cache" in categories
    cache_rec = [r for r in recs if r.category == "cache"][0]
    assert cache_rec.severity == "critical"


def test_kv_cache_warning():
    snap = _make_snapshot(gpu_cache_usage_percent=0.85)
    recs = analyze(snap)
    cache_recs = [r for r in recs if r.category == "cache"]
    assert len(cache_recs) == 1
    assert cache_recs[0].severity == "warning"


def test_queue_backup():
    snap = _make_snapshot(requests_waiting=12)
    recs = analyze(snap)
    queue_recs = [r for r in recs if r.title == "Request queue backup"]
    assert len(queue_recs) == 1
    assert queue_recs[0].severity == "critical"


def test_high_ttft():
    snap = _make_snapshot(time_to_first_token_p95=0.6)  # 600ms
    recs = analyze(snap)
    ttft_recs = [r for r in recs if "time-to-first-token" in r.title.lower()]
    assert len(ttft_recs) == 1
    assert ttft_recs[0].severity == "critical"


def test_batch_full_with_queue():
    snap = _make_snapshot(
        avg_batch_size=15.0,
        max_batch_size=16,
        requests_waiting=5,
    )
    recs = analyze(snap)
    batch_recs = [r for r in recs if r.category == "batching"]
    assert len(batch_recs) == 1
    assert "full" in batch_recs[0].title.lower()


def test_low_batch_no_queue():
    snap = _make_snapshot(
        avg_batch_size=3.0,
        max_batch_size=16,
        requests_waiting=0,
    )
    recs = analyze(snap)
    batch_recs = [r for r in recs if r.category == "batching"]
    assert len(batch_recs) == 1
    assert batch_recs[0].severity == "info"


def test_gpu_underutilization_only_with_nvml_data():
    """Should not flag underutilization when GPU fields are all zeros (no NVML)."""
    snap = _make_snapshot(
        gpu_utilization_percent=0,
        gpu_memory_total_gb=0,
        gpu_memory_used_gb=0,
    )
    recs = analyze(snap)
    util_recs = [r for r in recs if r.category == "utilization"]
    assert len(util_recs) == 0


def test_gpu_underutilization_with_nvml_data():
    """Should flag when we have real GPU data showing low utilization."""
    snap = _make_snapshot(
        gpu_utilization_percent=0.15,
        gpu_memory_total_gb=24.0,
        gpu_memory_used_gb=10.0,
    )
    recs = analyze(snap)
    util_recs = [r for r in recs if r.category == "utilization"]
    assert len(util_recs) == 1


def test_recommendations_sorted_by_severity():
    snap = _make_snapshot(
        gpu_cache_usage_percent=0.95,  # critical
        requests_waiting=7,             # warning
        avg_batch_size=3.0,             # info
        max_batch_size=16,
    )
    recs = analyze(snap)
    assert len(recs) >= 2
    assert recs[0].severity == "critical"


def test_latency_trend_detection():
    # Build history where TTFT was low, then current is much higher
    old_snaps = [
        _make_snapshot(time_to_first_token_p95=0.08) for _ in range(10)
    ]
    current = _make_snapshot(time_to_first_token_p95=0.20)  # 150% increase

    recs = analyze(current, history=old_snaps)
    trend_recs = [r for r in recs if "trending" in r.title.lower()]
    assert len(trend_recs) == 1


def test_config_hints_present():
    snap = _make_snapshot(gpu_cache_usage_percent=0.95)
    recs = analyze(snap)
    cache_recs = [r for r in recs if r.category == "cache"]
    assert cache_recs[0].config_hint is not None
