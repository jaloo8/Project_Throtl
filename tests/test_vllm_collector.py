"""
Tests for the vLLM collector using the fake metrics server.

Starts the fake server in a thread, points the collector at it,
and verifies we get reasonable InferenceSnapshots back.
"""

import threading
import time
from http.server import HTTPServer

from src.throtl.collector.vllm_collector import VLLMCollector
from src.throtl.mock.fake_vllm_server import _MetricsHandler


def _start_test_server(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)  # let it bind
    return server


def test_collector_gets_snapshot_from_fake_server():
    server = _start_test_server(19876)
    try:
        collector = VLLMCollector(base_url="http://127.0.0.1:19876")
        snap = collector.collect()

        assert snap.requests_running >= 1
        assert snap.tokens_per_second > 0
        assert 0 <= snap.gpu_cache_usage_percent <= 1.0
        assert snap.prompt_tokens_total > 0
        assert snap.generation_tokens_total > 0
        assert snap.time_to_first_token_p50 >= 0
        assert snap.estimated_cost_per_1k_tokens > 0

        collector.close()
    finally:
        server.shutdown()


def test_collector_accumulates_across_calls():
    server = _start_test_server(19877)
    try:
        collector = VLLMCollector(base_url="http://127.0.0.1:19877")
        snap1 = collector.collect()
        snap2 = collector.collect()

        assert snap2.generation_tokens_total >= snap1.generation_tokens_total
        assert snap2.prompt_tokens_total >= snap1.prompt_tokens_total

        collector.close()
    finally:
        server.shutdown()


def test_collector_name_includes_url():
    collector = VLLMCollector(base_url="http://localhost:8000")
    assert "localhost:8000" in collector.name()
    collector.close()
