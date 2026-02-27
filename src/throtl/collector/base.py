"""
Base collector interface. Anything that can produce an InferenceSnapshot
implements this so the dashboard doesn't care where data comes from.
"""

from abc import ABC, abstractmethod

from throtl.metrics import InferenceSnapshot


class MetricsCollector(ABC):

    @abstractmethod
    def collect(self) -> InferenceSnapshot:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
