"""Tests for SQLite metrics storage."""

import os
import tempfile
from datetime import datetime, timezone

from throtl.metrics import InferenceSnapshot
from throtl.storage.sqlite_store import MetricsStore


def _make_snapshot(**overrides) -> InferenceSnapshot:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
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


def test_savings_ledger():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MetricsStore(db_path=db_path)

        store.record_savings(
            description="Increased --max-num-seqs from 16 to 32",
            cost_before=0.005,
            cost_after=0.003,
            config_change="--max-num-seqs 32",
        )
        store.record_savings(
            description="Enabled prefix caching",
            cost_before=0.003,
            cost_after=0.002,
        )

        history = store.get_savings_history()
        assert len(history) == 2
        assert "prefix" in history[1]["description"].lower()

        total = store.total_estimated_daily_savings()
        assert total > 0

        store.close()
    finally:
        os.unlink(db_path)


def test_tuning_events():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MetricsStore(db_path=db_path)

        store.record_tuning_event(
            suggestion_title="Increase batch size",
            config_flag="--max-num-seqs",
            config_change="16 -> 32",
            ttft_p95_before=0.25,
            ttft_p95_after=0.15,
            tps_before=80.0,
            tps_after=120.0,
            cost_per_1k_before=0.005,
            cost_per_1k_after=0.003,
            outcome="improved",
        )

        events = store.get_tuning_history()
        assert len(events) == 1
        assert events[0]["suggestion_title"] == "Increase batch size"
        assert events[0]["outcome"] == "improved"

        store.close()
    finally:
        os.unlink(db_path)


def test_schema_migration_from_v0():
    """Opening an old DB without new columns should auto-migrate."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        conn = __import__("sqlite3").connect(db_path)
        conn.execute("""
            CREATE TABLE snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                requests_running INTEGER,
                requests_waiting INTEGER,
                requests_completed INTEGER,
                prompt_tokens_total INTEGER,
                generation_tokens_total INTEGER,
                tokens_per_second REAL,
                ttft_p50 REAL, ttft_p95 REAL, ttft_p99 REAL,
                tbt_p50 REAL, tbt_p95 REAL, tbt_p99 REAL,
                gpu_cache_usage_pct REAL,
                gpu_memory_used_gb REAL, gpu_memory_total_gb REAL,
                gpu_utilization_pct REAL,
                avg_batch_size REAL, max_batch_size INTEGER,
                cost_per_1k_tokens REAL
            )
        """)
        conn.commit()
        conn.close()

        store = MetricsStore(db_path=db_path)
        store.save(_make_snapshot(cumulative_cost_usd=5.0, sla_compliance_percent=0.95))

        loaded = store.get_recent(minutes=10)
        assert len(loaded) == 1
        assert abs(loaded[0].cumulative_cost_usd - 5.0) < 0.01
        assert abs(loaded[0].sla_compliance_percent - 0.95) < 0.01

        store.close()
    finally:
        os.unlink(db_path)


def test_cost_fields_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = MetricsStore(db_path=db_path)

        original = _make_snapshot(
            gpu_cost_per_hour=2.50,
            cumulative_cost_usd=15.75,
            sla_target_ttft_ms=150.0,
            sla_compliance_percent=0.92,
        )
        store.save(original)

        loaded = store.get_recent(minutes=10)
        snap = loaded[0]
        assert abs(snap.gpu_cost_per_hour - 2.50) < 0.01
        assert abs(snap.cumulative_cost_usd - 15.75) < 0.01
        assert abs(snap.sla_target_ttft_ms - 150.0) < 0.1
        assert abs(snap.sla_compliance_percent - 0.92) < 0.01

        store.close()
    finally:
        os.unlink(db_path)
