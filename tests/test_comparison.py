"""Tests for before/after comparison engine."""

from datetime import datetime, timezone

from throtl.engine.comparison import compare
from throtl.metrics import InferenceSnapshot


def _snap(**overrides) -> InferenceSnapshot:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        requests_running=8,
        requests_waiting=2,
        tokens_per_second=80.0,
        time_to_first_token_p50=0.06,
        time_to_first_token_p95=0.12,
        time_to_first_token_p99=0.20,
        time_per_output_token_p50=0.015,
        time_per_output_token_p95=0.025,
        gpu_cache_usage_percent=0.50,
        gpu_utilization_percent=0.65,
        avg_batch_size=8.0,
        estimated_cost_per_1k_tokens=0.003,
    )
    defaults.update(overrides)
    return InferenceSnapshot(**defaults)


def test_no_change_detected():
    before = [_snap() for _ in range(5)]
    after = [_snap() for _ in range(5)]
    report = compare(before, after)
    assert "No significant changes" in report.summary


def test_improvement_detected():
    before = [_snap(tokens_per_second=60.0, time_to_first_token_p95=0.25)]
    after = [_snap(tokens_per_second=100.0, time_to_first_token_p95=0.08)]
    report = compare(before, after)
    assert len(report.improvements) > 0
    assert report.before_count == 1
    assert report.after_count == 1


def test_regression_detected():
    before = [_snap(tokens_per_second=100.0)]
    after = [_snap(tokens_per_second=50.0)]
    report = compare(before, after)
    assert len(report.regressions) > 0


def test_mixed_results():
    # Better throughput but worse latency
    before = [_snap(tokens_per_second=60.0, time_to_first_token_p95=0.10)]
    after = [_snap(tokens_per_second=100.0, time_to_first_token_p95=0.30)]
    report = compare(before, after)
    assert "improved" in report.summary
    assert "regressed" in report.summary


def test_empty_windows():
    report = compare([], [_snap()])
    assert "Need snapshots" in report.summary


def test_delta_pct_calculated():
    before = [_snap(tokens_per_second=100.0)]
    after = [_snap(tokens_per_second=120.0)]
    report = compare(before, after)
    tps_delta = [d for d in report.deltas if d.name == "Tokens/sec"][0]
    assert abs(tps_delta.delta_pct - 20.0) < 0.1
