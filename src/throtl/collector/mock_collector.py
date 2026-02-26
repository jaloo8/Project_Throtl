"""Collector backed by the mock generator. No GPU needed."""

from src.throtl.collector.base import MetricsCollector
from src.throtl.metrics import InferenceSnapshot
from src.throtl.mock.generator import MockVLLMServer


class MockCollector(MetricsCollector):

    def __init__(self, seed: int = 42):
        self._server = MockVLLMServer(seed=seed)

    def collect(self) -> InferenceSnapshot:
        return self._server.snapshot()

    def name(self) -> str:
        return "Mock vLLM (Llama 3 8B, simulated)"
