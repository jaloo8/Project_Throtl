"""
Optional GPU stats via NVML (the library behind nvidia-smi).

Falls back gracefully when pynvml isn't installed or no GPU is present,
so the rest of the app works fine on machines without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


@dataclass
class GPUStats:
    utilization_percent: float   # 0.0 - 1.0
    memory_used_gb: float
    memory_total_gb: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float


class GPUMonitor:
    """Reads GPU metrics from NVML. Safe to construct even without a GPU."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._handle = None
        self._initialized = False

        if not _NVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._initialized = True
        except pynvml.NVMLError:
            pass

    @property
    def available(self) -> bool:
        return self._initialized

    def read(self) -> Optional[GPUStats]:
        """Read current GPU stats. Returns None if NVML isn't available."""
        if not self._initialized:
            return None

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle)       # milliwatts
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)

            return GPUStats(
                utilization_percent=util.gpu / 100.0,
                memory_used_gb=mem.used / (1024 ** 3),
                memory_total_gb=mem.total / (1024 ** 3),
                temperature_c=temp,
                power_draw_w=power / 1000.0,
                power_limit_w=power_limit / 1000.0,
            )
        except pynvml.NVMLError:
            return None

    def close(self):
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._initialized = False
