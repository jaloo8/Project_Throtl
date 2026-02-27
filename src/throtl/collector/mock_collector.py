"""Collector backed by the mock generator. No GPU needed."""

from throtl.collector.base import MetricsCollector
from throtl.metrics import InferenceSnapshot
from throtl.mock.generator import MockVLLMServer


class MockCollector(MetricsCollector):

    def __init__(self, seed: int = 42, gpu_cost_per_hour: float = 1.0):
        self._server = MockVLLMServer(seed=seed, gpu_cost_per_hour=gpu_cost_per_hour)

    def collect(self) -> InferenceSnapshot:
        return self._server.snapshot()

    def name(self) -> str:
        return "Mock vLLM (Llama 3 8B, simulated)"
