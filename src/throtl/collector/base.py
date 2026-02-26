"""
Base collector interface.

A collector is anything that can produce an InferenceSnapshot.
This keeps the dashboard and storage layers decoupled from where
the data actually comes from (mock, real vLLM, Triton, etc).
"""

from abc import ABC, abstractmethod

from src.throtl.metrics import InferenceSnapshot


class MetricsCollector(ABC):
    """Interface for all metrics sources."""

    @abstractmethod
    def collect(self) -> InferenceSnapshot:
        """Fetch one snapshot of current metrics."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this source."""
        ...
