"""
Collector that reads from the mock vLLM server.
Used for local development on machines without a GPU.
"""

from src.throtl.collector.base import MetricsCollector
from src.throtl.metrics import InferenceSnapshot
from src.throtl.mock.generator import MockVLLMServer


class MockCollector(MetricsCollector):
    """Wraps the mock generator as a standard collector."""

    def __init__(self, seed: int = 42):
        self._server = MockVLLMServer(seed=seed)

    def collect(self) -> InferenceSnapshot:
        return self._server.snapshot()

    def name(self) -> str:
        return "Mock vLLM (Llama 3 8B on simulated RTX 4090)"
