"""Tests for the Prometheus text format parser."""

from src.throtl.collector.prometheus_parser import (
    get_counter,
    get_gauge,
    get_histogram_percentile,
    parse_labels,
    parse_prometheus_text,
)

SAMPLE_VLLM_OUTPUT = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 12

# HELP vllm:num_requests_waiting Number of requests waiting to be processed
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 3

# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.7300

# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 148329

# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 52841

# HELP vllm:avg_prompt_throughput_toks_per_s Average prefill throughput
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s 45.2

# HELP vllm:avg_generation_throughput_toks_per_s Average generation throughput
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s 68.7

# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.01"} 5
vllm:time_to_first_token_seconds_bucket{le="0.02"} 15
vllm:time_to_first_token_seconds_bucket{le="0.04"} 40
vllm:time_to_first_token_seconds_bucket{le="0.06"} 70
vllm:time_to_first_token_seconds_bucket{le="0.08"} 85
vllm:time_to_first_token_seconds_bucket{le="0.1"} 92
vllm:time_to_first_token_seconds_bucket{le="0.25"} 98
vllm:time_to_first_token_seconds_bucket{le="0.5"} 100
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100
vllm:time_to_first_token_seconds_count 100
vllm:time_to_first_token_seconds_sum 4.5600
"""


def test_parse_labels():
    result = parse_labels('model_name="llama",le="0.5"')
    assert result == {"model_name": "llama", "le": "0.5"}


def test_parse_labels_empty():
    assert parse_labels("") == {}


def test_parse_gauges():
    families = parse_prometheus_text(SAMPLE_VLLM_OUTPUT)
    assert get_gauge(families, "vllm:num_requests_running") == 12
    assert get_gauge(families, "vllm:num_requests_waiting") == 3
    assert get_gauge(families, "vllm:gpu_cache_usage_perc") == 0.73


def test_parse_counters():
    families = parse_prometheus_text(SAMPLE_VLLM_OUTPUT)
    assert get_counter(families, "vllm:prompt_tokens") == 148329
    assert get_counter(families, "vllm:generation_tokens") == 52841


def test_parse_throughput_gauges():
    families = parse_prometheus_text(SAMPLE_VLLM_OUTPUT)
    assert get_gauge(families, "vllm:avg_prompt_throughput_toks_per_s") == 45.2
    assert get_gauge(families, "vllm:avg_generation_throughput_toks_per_s") == 68.7


def test_histogram_percentile_p50():
    families = parse_prometheus_text(SAMPLE_VLLM_OUTPUT)
    p50 = get_histogram_percentile(families, "vllm:time_to_first_token_seconds", 0.50)
    assert p50 is not None
    # p50 should be somewhere around 0.04-0.06 based on the bucket distribution
    assert 0.02 < p50 < 0.08


def test_histogram_percentile_p95():
    families = parse_prometheus_text(SAMPLE_VLLM_OUTPUT)
    p95 = get_histogram_percentile(families, "vllm:time_to_first_token_seconds", 0.95)
    assert p95 is not None
    # p95 should be in the higher buckets
    assert 0.08 < p95 < 0.30


def test_missing_metric_returns_none():
    families = parse_prometheus_text(SAMPLE_VLLM_OUTPUT)
    assert get_gauge(families, "nonexistent_metric") is None
    assert get_counter(families, "nonexistent_metric") is None
    assert get_histogram_percentile(families, "nonexistent_metric", 0.5) is None


def test_empty_input():
    families = parse_prometheus_text("")
    assert len(families) == 0


def test_handles_comments_and_blank_lines():
    text = """
    # This is a comment
    # HELP my_gauge A test gauge
    # TYPE my_gauge gauge
    my_gauge 42.5

    """
    families = parse_prometheus_text(text)
    assert get_gauge(families, "my_gauge") == 42.5
