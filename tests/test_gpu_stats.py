"""Tests for GPUMonitor graceful fallback (no GPU on this machine)."""

from src.throtl.collector.gpu_stats import GPUMonitor


def test_gpu_monitor_unavailable_on_mac():
    """On a machine without an NVIDIA GPU, monitor should init safely."""
    monitor = GPUMonitor()
    assert monitor.available is False


def test_read_returns_none_without_gpu():
    monitor = GPUMonitor()
    assert monitor.read() is None


def test_close_is_safe_without_gpu():
    monitor = GPUMonitor()
    monitor.close()  # should not raise
    assert monitor.available is False
