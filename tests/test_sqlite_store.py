"""Tests for SQLite metrics storage."""

import os
import tempfile
from datetime import datetime

from src.throtl.metrics import InferenceSnapshot
from src.throtl.storage.sqlite_store import MetricsStore


def _make_snapshot(**overrides) -> InferenceSnapshot:
    defaults = dict(
        timestamp=datetime.now(),
        requests_running=5,
        requests_waiting=2,
        tokens_per_second=100.0,
        gpu_cache_usage_percent=0.65,
        estimated_cost_per_1k_tokens=0.003,
    )
    defaults.update(overrides)
    return InferenceSnapshot(**defaults)


def test_save_and_count():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MetricsStore(db_path=db_path)
        assert store.count() == 0

        store.save(_make_snapshot())
        store.save(_make_snapshot())
        store.save(_make_snapshot())

        assert store.count() == 3
        store.close()
    finally:
        os.unlink(db_path)


def test_get_recent():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MetricsStore(db_path=db_path)

        for i in range(5):
            store.save(_make_snapshot(requests_running=i + 1))

        recent = store.get_recent(minutes=10)
        assert len(recent) == 5
        assert recent[0].requests_running == 1
        assert recent[4].requests_running == 5
        store.close()
    finally:
        os.unlink(db_path)


def test_roundtrip_preserves_values():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MetricsStore(db_path=db_path)

        original = _make_snapshot(
            requests_running=8,
            requests_waiting=3,
            tokens_per_second=142.5,
            time_to_first_token_p50=0.045,
            time_to_first_token_p95=0.12,
            gpu_cache_usage_percent=0.78,
            estimated_cost_per_1k_tokens=0.0025,
        )
        store.save(original)

        loaded = store.get_recent(minutes=10)
        assert len(loaded) == 1
        snap = loaded[0]

        assert snap.requests_running == 8
        assert snap.requests_waiting == 3
        assert abs(snap.tokens_per_second - 142.5) < 0.01
        assert abs(snap.time_to_first_token_p50 - 0.045) < 0.001
        assert abs(snap.gpu_cache_usage_percent - 0.78) < 0.01
        store.close()
    finally:
        os.unlink(db_path)
