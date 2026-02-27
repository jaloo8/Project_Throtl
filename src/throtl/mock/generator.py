"""
Mock vLLM metrics generator.

Produces fake but realistic metrics so we can develop and test without
a GPU. Numbers are loosely based on Llama 3 8B under moderate traffic
with a 24GB GPU.
"""

import math
import random
from datetime import datetime, timezone

from throtl.metrics import InferenceSnapshot


class MockVLLMServer:

    def __init__(self, seed: int = 42, gpu_cost_per_hour: float = 1.0):
        self._rng = random.Random(seed)
        self._tick = 0
        self._total_prompt_tokens = 0
        self._total_gen_tokens = 0
        self._total_completed = 0
        self._cumulative_cost = 0.0
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.gpu_memory_total_gb = 24.0
        self.model_memory_gb = 8.5  # ~8B params in fp16
        self.sla_target_ttft_ms = 200.0

    def snapshot(self) -> InferenceSnapshot:
        """Generate one reading, advancing the simulation clock."""
        self._tick += 1
        t = self._tick

        # Sinusoidal base load with occasional random spikes
        base_load = 8 + 6 * math.sin(t * 0.05)
        spike = self._rng.random() * 8 if self._rng.random() > 0.9 else 0
        active_requests = max(1, int(base_load + spike))

        # Queue starts building once we exceed ~12 concurrent requests
        queue_pressure = max(0, active_requests - 12)
        waiting = int(queue_pressure * self._rng.uniform(0.5, 1.5))

        # Throughput scales with batch size (capped at 16)
        batch_size = min(active_requests, 16)
        base_tps = 45 + batch_size * 8
        tokens_per_second = max(10, base_tps + self._rng.gauss(0, 5))

        gen_this_tick = int(tokens_per_second * 2)
        prompt_this_tick = int(gen_this_tick * 0.6)
        self._total_gen_tokens += gen_this_tick
        self._total_prompt_tokens += prompt_this_tick
        self._total_completed += max(1, active_requests // 3)

        # KV cache grows with active sequences
        cache_base = 0.3 + (active_requests / 20) * 0.5
        cache_usage = max(0.1, min(0.98, cache_base + self._rng.gauss(0, 0.03)))

        # VRAM = model weights + KV cache
        kv_cache_gb = (self.gpu_memory_total_gb - self.model_memory_gb) * cache_usage
        gpu_mem_used = self.model_memory_gb + kv_cache_gb
        gpu_util = min(0.99, 0.3 + (tokens_per_second / 200) + self._rng.gauss(0, 0.03))

        # TTFT degrades with queue depth and cache pressure
        ttft_base = 0.08 + (waiting * 0.02) + (cache_usage * 0.05)
        ttft_p50 = max(0.02, ttft_base + self._rng.gauss(0, 0.01))
        ttft_p95 = ttft_p50 * self._rng.uniform(1.8, 2.5)
        ttft_p99 = ttft_p95 * self._rng.uniform(1.3, 1.8)

        # TBT is more stable -- mostly depends on how busy the GPU is
        tbt_base = 0.012 + (active_requests / 50) * 0.005
        tbt_p50 = max(0.005, tbt_base + self._rng.gauss(0, 0.001))
        tbt_p95 = tbt_p50 * self._rng.uniform(1.5, 2.0)
        tbt_p99 = tbt_p95 * self._rng.uniform(1.2, 1.5)

        # Cost: configurable GPU rate spread across tokens generated
        tokens_per_hour = tokens_per_second * 3600
        cost_per_1k = (self.gpu_cost_per_hour / max(1, tokens_per_hour)) * 1000

        # Each tick represents ~2 seconds of wall time
        tick_cost = self.gpu_cost_per_hour / 1800
        self._cumulative_cost += tick_cost

        # SLA compliance: approximate fraction of requests meeting TTFT target
        target_s = self.sla_target_ttft_ms / 1000.0
        sla_compliance = max(0.0, min(1.0, 1.0 - (ttft_p95 / target_s) * 0.3))

        return InferenceSnapshot(
            timestamp=datetime.now(timezone.utc),
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
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            estimated_cost_per_1k_tokens=cost_per_1k,
            cumulative_cost_usd=self._cumulative_cost,
            sla_target_ttft_ms=self.sla_target_ttft_ms,
            sla_compliance_percent=sla_compliance,
        )
