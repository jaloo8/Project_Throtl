"""
Metric definitions that mirror vLLM's /metrics endpoint (Prometheus format),
plus GPU-level signals we'll add when running on real hardware.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class InferenceSnapshot:
    """One point-in-time reading from an inference server."""

    timestamp: datetime

    # Requests
    requests_running: int = 0
    requests_waiting: int = 0
    requests_completed: int = 0

    # Throughput
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    tokens_per_second: float = 0.0

    # Latency (in seconds, converted to ms for display)
    time_to_first_token_p50: float = 0.0
    time_to_first_token_p95: float = 0.0
    time_to_first_token_p99: float = 0.0
    time_per_output_token_p50: float = 0.0
    time_per_output_token_p95: float = 0.0
    time_per_output_token_p99: float = 0.0

    # GPU / KV cache
    gpu_cache_usage_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization_percent: float = 0.0

    # Batching
    avg_batch_size: float = 0.0
    max_batch_size: int = 0

    # Rough cost estimate based on GPU hourly rate
    estimated_cost_per_1k_tokens: float = 0.0

    def summary(self) -> dict:
        """Flat dict with display-friendly units (ms, %, etc)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "requests_running": self.requests_running,
            "requests_waiting": self.requests_waiting,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "ttft_p50_ms": round(self.time_to_first_token_p50 * 1000, 1),
            "ttft_p95_ms": round(self.time_to_first_token_p95 * 1000, 1),
            "tbt_p50_ms": round(self.time_per_output_token_p50 * 1000, 1),
            "gpu_cache_pct": round(self.gpu_cache_usage_percent * 100, 1),
            "gpu_util_pct": round(self.gpu_utilization_percent * 100, 1),
            "avg_batch_size": round(self.avg_batch_size, 1),
            "cost_per_1k_tokens": round(self.estimated_cost_per_1k_tokens, 4),
        }
