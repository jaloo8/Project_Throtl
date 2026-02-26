"""Basic sanity checks for the mock metrics generator."""

from src.throtl.mock.generator import MockVLLMServer


def test_snapshot_returns_valid_data():
    server = MockVLLMServer(seed=42)
    snap = server.snapshot()

    assert snap.requests_running >= 1
    assert snap.tokens_per_second > 0
    assert 0 <= snap.gpu_cache_usage_percent <= 1.0
    assert 0 <= snap.gpu_utilization_percent <= 1.0
    assert snap.gpu_memory_used_gb <= snap.gpu_memory_total_gb
    assert snap.time_to_first_token_p50 > 0
    assert snap.time_per_output_token_p50 > 0
    assert snap.estimated_cost_per_1k_tokens > 0


def test_snapshots_accumulate_tokens():
    server = MockVLLMServer(seed=42)
    snap1 = server.snapshot()
    snap2 = server.snapshot()

    assert snap2.generation_tokens_total > snap1.generation_tokens_total
    assert snap2.prompt_tokens_total > snap1.prompt_tokens_total
    assert snap2.requests_completed >= snap1.requests_completed


def test_deterministic_with_same_seed():
    server_a = MockVLLMServer(seed=99)
    server_b = MockVLLMServer(seed=99)

    snap_a = server_a.snapshot()
    snap_b = server_b.snapshot()

    assert snap_a.requests_running == snap_b.requests_running
    assert snap_a.tokens_per_second == snap_b.tokens_per_second
    assert snap_a.gpu_cache_usage_percent == snap_b.gpu_cache_usage_percent


def test_summary_dict_has_expected_keys():
    server = MockVLLMServer(seed=42)
    summary = server.snapshot().summary()

    expected_keys = [
        "timestamp", "requests_running", "requests_waiting",
        "tokens_per_second", "ttft_p50_ms", "ttft_p95_ms",
        "tbt_p50_ms", "gpu_cache_pct", "gpu_util_pct",
        "avg_batch_size", "cost_per_1k_tokens",
    ]
    for key in expected_keys:
        assert key in summary, f"Missing key: {key}"
