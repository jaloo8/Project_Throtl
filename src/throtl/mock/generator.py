"""
Mock vLLM metrics generator.

Produces realistic-looking inference metrics so we can build and test
the collector and dashboard without needing a real GPU or vLLM instance.

The numbers are modeled loosely on a Llama 3 8B running on a single
RTX 4090 (24GB VRAM) under moderate traffic. They drift over time to
simulate real-world patterns: traffic ramps, cache pressure, queue
buildups, and occasional latency spikes.
"""

import math
import random
from datetime import datetime

from src.throtl.metrics import InferenceSnapshot


class MockVLLMServer:
    """Simulates a vLLM server's metric output over time."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._tick = 0
        self._total_prompt_tokens = 0
        self._total_gen_tokens = 0
        self._total_completed = 0

        # "Hardware" config for our simulated GPU
        self.gpu_memory_total_gb = 24.0
        self.model_memory_gb = 8.5  # ~8B param model in fp16

    def snapshot(self) -> InferenceSnapshot:
        """Generate one metrics reading, advancing the simulation clock."""
        self._tick += 1
        t = self._tick

        # Traffic pattern: sinusoidal base load + random spikes
        base_load = 8 + 6 * math.sin(t * 0.05)
        spike = self._rng.random() * 8 if self._rng.random() > 0.9 else 0
        active_requests = max(1, int(base_load + spike))

        # Queue builds up when load is high
        queue_pressure = max(0, active_requests - 12)
        waiting = int(queue_pressure * self._rng.uniform(0.5, 1.5))

        # Tokens per second depends on batch size and GPU headroom
        batch_size = min(active_requests, 16)
        base_tps = 45 + batch_size * 8
        noise = self._rng.gauss(0, 5)
        tokens_per_second = max(10, base_tps + noise)

        # Accumulate totals
        gen_this_tick = int(tokens_per_second * 2)
        prompt_this_tick = int(gen_this_tick * 0.6)
        self._total_gen_tokens += gen_this_tick
        self._total_prompt_tokens += prompt_this_tick
        self._total_completed += max(1, active_requests // 3)

        # KV cache usage correlates with active sequences and context length
        cache_base = 0.3 + (active_requests / 20) * 0.5
        cache_noise = self._rng.gauss(0, 0.03)
        cache_usage = max(0.1, min(0.98, cache_base + cache_noise))

        # GPU memory: model + KV cache
        kv_cache_gb = (self.gpu_memory_total_gb - self.model_memory_gb) * cache_usage
        gpu_mem_used = self.model_memory_gb + kv_cache_gb

        # GPU utilization tracks with throughput
        gpu_util = min(0.99, 0.3 + (tokens_per_second / 200) + self._rng.gauss(0, 0.03))

        # Latency: TTFT gets worse under load, TBT is more stable
        ttft_base = 0.08 + (waiting * 0.02) + (cache_usage * 0.05)
        ttft_p50 = max(0.02, ttft_base + self._rng.gauss(0, 0.01))
        ttft_p95 = ttft_p50 * self._rng.uniform(1.8, 2.5)
        ttft_p99 = ttft_p95 * self._rng.uniform(1.3, 1.8)

        tbt_base = 0.012 + (active_requests / 50) * 0.005
        tbt_p50 = max(0.005, tbt_base + self._rng.gauss(0, 0.001))
        tbt_p95 = tbt_p50 * self._rng.uniform(1.5, 2.0)
        tbt_p99 = tbt_p95 * self._rng.uniform(1.2, 1.5)

        # Rough cost estimate: assume $1/hr GPU cost, spread across tokens
        hourly_cost = 1.0
        tokens_per_hour = tokens_per_second * 3600
        cost_per_1k = (hourly_cost / max(1, tokens_per_hour)) * 1000

        return InferenceSnapshot(
            timestamp=datetime.now(),
            requests_running=active_requests,
            requests_waiting=waiting,
            requests_completed=self._total_completed,
            prompt_tokens_total=self._total_prompt_tokens,
            generation_tokens_total=self._total_gen_tokens,
            tokens_per_second=tokens_per_second,
            time_to_first_token_p50=ttft_p50,
            time_to_first_token_p95=ttft_p95,
            time_to_first_token_p99=ttft_p99,
            time_per_output_token_p50=tbt_p50,
            time_per_output_token_p95=tbt_p95,
            time_per_output_token_p99=tbt_p99,
            gpu_cache_usage_percent=cache_usage,
            gpu_memory_used_gb=gpu_mem_used,
            gpu_memory_total_gb=self.gpu_memory_total_gb,
            gpu_utilization_percent=gpu_util,
            avg_batch_size=float(batch_size),
            max_batch_size=16,
            estimated_cost_per_1k_tokens=cost_per_1k,
        )
