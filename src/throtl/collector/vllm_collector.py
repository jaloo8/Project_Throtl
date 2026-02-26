"""
Collector for a live vLLM server. Scrapes /metrics and maps
the Prometheus output into InferenceSnapshot. Missing metrics
(which vary by vLLM version) default to zero.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

from src.throtl.collector.base import MetricsCollector
from src.throtl.collector.prometheus_parser import (
    get_counter,
    get_gauge,
    get_histogram_percentile,
    parse_prometheus_text,
)
from src.throtl.metrics import InferenceSnapshot


class VLLMCollector(MetricsCollector):

    def __init__(
        self,
        base_url: str,
        gpu_cost_per_hour: float = 1.0,
        timeout_seconds: float = 5.0,
    ):
        self._metrics_url = base_url.rstrip("/")
        if not self._metrics_url.endswith("/metrics"):
            self._metrics_url += "/metrics"

        self._gpu_cost_per_hour = gpu_cost_per_hour
        self._timeout = timeout_seconds
        self._client = httpx.Client(timeout=self._timeout)
        self._prev_prompt_tokens: Optional[float] = None
        self._prev_gen_tokens: Optional[float] = None

    def collect(self) -> InferenceSnapshot:
        """Scrape /metrics, parse the Prometheus text, return a snapshot."""
        response = self._client.get(self._metrics_url)
        response.raise_for_status()

        families = parse_prometheus_text(response.text)

        requests_running = int(get_gauge(families, "vllm:num_requests_running") or 0)
        requests_waiting = int(get_gauge(families, "vllm:num_requests_waiting") or 0)

        prompt_tokens = get_counter(families, "vllm:prompt_tokens") or 0
        gen_tokens = get_counter(families, "vllm:generation_tokens") or 0

        prompt_throughput = get_gauge(families, "vllm:avg_prompt_throughput_toks_per_s") or 0
        gen_throughput = get_gauge(families, "vllm:avg_generation_throughput_toks_per_s") or 0
        tokens_per_second = prompt_throughput + gen_throughput

        cache_usage = get_gauge(families, "vllm:gpu_cache_usage_perc") or 0

        ttft_p50 = get_histogram_percentile(families, "vllm:time_to_first_token_seconds", 0.50) or 0
        ttft_p95 = get_histogram_percentile(families, "vllm:time_to_first_token_seconds", 0.95) or 0
        ttft_p99 = get_histogram_percentile(families, "vllm:time_to_first_token_seconds", 0.99) or 0

        tbt_p50 = get_histogram_percentile(families, "vllm:time_per_output_token_seconds", 0.50) or 0
        tbt_p95 = get_histogram_percentile(families, "vllm:time_per_output_token_seconds", 0.95) or 0
        tbt_p99 = get_histogram_percentile(families, "vllm:time_per_output_token_seconds", 0.99) or 0

        # vLLM doesn't expose batch size or GPU utilization directly --
        # we estimate batch from running requests and leave GPU util at 0
        # until we add nvidia-smi / NVML integration.
        avg_batch_size = float(requests_running)
        # Rough max batch size guess -- will be configurable later
        max_batch_size = 16

        # Cost estimate
        tokens_per_hour = max(1, tokens_per_second * 3600)
        cost_per_1k = (self._gpu_cost_per_hour / tokens_per_hour) * 1000

        return InferenceSnapshot(
            timestamp=datetime.now(),
            requests_running=requests_running,
            requests_waiting=requests_waiting,
            requests_completed=0,  # vLLM doesn't expose a simple completed counter
            prompt_tokens_total=int(prompt_tokens),
            generation_tokens_total=int(gen_tokens),
            tokens_per_second=tokens_per_second,
            time_to_first_token_p50=ttft_p50,
            time_to_first_token_p95=ttft_p95,
            time_to_first_token_p99=ttft_p99,
            time_per_output_token_p50=tbt_p50,
            time_per_output_token_p95=tbt_p95,
            time_per_output_token_p99=tbt_p99,
            gpu_cache_usage_percent=cache_usage,
            gpu_memory_used_gb=0,  # needs NVML, not available from /metrics
            gpu_memory_total_gb=0,
            gpu_utilization_percent=0,
            avg_batch_size=avg_batch_size,
            max_batch_size=max_batch_size,
            estimated_cost_per_1k_tokens=cost_per_1k,
        )

    def name(self) -> str:
        return f"vLLM ({self._metrics_url})"

    def close(self):
        self._client.close()
