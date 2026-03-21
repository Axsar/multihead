"""Tests for observability module."""

from __future__ import annotations

import json
import logging

from multihead.observability import MetricsCollector, StructuredLogger, Timer


class TestMetricsCollector:
    def test_counter_inc(self):
        m = MetricsCollector()
        m.inc("requests")
        m.inc("requests")
        m.inc("requests", 3.0)
        assert m.counter("requests") == 5.0

    def test_counter_with_labels(self):
        m = MetricsCollector()
        m.inc("http_requests", labels={"method": "GET"})
        m.inc("http_requests", labels={"method": "POST"})
        m.inc("http_requests", labels={"method": "GET"})
        assert m.counter("http_requests", labels={"method": "GET"}) == 2.0
        assert m.counter("http_requests", labels={"method": "POST"}) == 1.0

    def test_gauge_set(self):
        m = MetricsCollector()
        m.set_gauge("vram_free", 8192.0)
        assert m.gauge("vram_free") == 8192.0
        m.set_gauge("vram_free", 4096.0)
        assert m.gauge("vram_free") == 4096.0

    def test_gauge_default(self):
        m = MetricsCollector()
        assert m.gauge("nonexistent") == 0.0

    def test_histogram(self):
        m = MetricsCollector()
        m.observe("latency", 0.1)
        m.observe("latency", 0.2)
        m.observe("latency", 0.3)
        stats = m.histogram("latency")
        assert stats["count"] == 3
        assert abs(stats["avg"] - 0.2) < 1e-9
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3

    def test_histogram_empty(self):
        m = MetricsCollector()
        stats = m.histogram("empty")
        assert stats["count"] == 0

    def test_to_json(self):
        m = MetricsCollector()
        m.inc("reqs", 5.0)
        m.set_gauge("mem", 1024.0)
        m.observe("lat", 0.5)
        data = m.to_json()
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data
        assert data["counters"]["reqs"]["value"] == 5.0
        assert data["gauges"]["mem"]["value"] == 1024.0
        assert data["histograms"]["lat"]["count"] == 1

    def test_to_prometheus(self):
        m = MetricsCollector()
        m.inc("requests_total", 10.0)
        m.set_gauge("vram_mb", 8192.0)
        text = m.to_prometheus()
        assert "requests_total" in text
        assert "vram_mb" in text
        assert "# TYPE" in text

    def test_prometheus_with_labels(self):
        m = MetricsCollector()
        m.inc("http_requests", 3.0, labels={"method": "GET"})
        text = m.to_prometheus()
        assert 'method="GET"' in text

    def test_reset(self):
        m = MetricsCollector()
        m.inc("a")
        m.set_gauge("b", 1.0)
        m.observe("c", 0.5)
        m.reset()
        assert m.counter("a") == 0.0
        assert m.gauge("b") == 0.0
        assert m.histogram("c")["count"] == 0


class TestStructuredLogger:
    def test_info_log(self, caplog):
        logger = StructuredLogger("test", level=logging.DEBUG)
        with caplog.at_level(logging.INFO, logger="test"):
            logger.info("hello", key="value")
        assert "hello" in caplog.text
        # Check JSON structure
        record = json.loads(caplog.records[0].message)
        assert record["msg"] == "hello"
        assert record["key"] == "value"
        assert "ts" in record
        assert record["level"] == "INFO"

    def test_context(self, caplog):
        logger = StructuredLogger("test_ctx", level=logging.DEBUG)
        logger.set_context(node_id="n1")
        with caplog.at_level(logging.INFO, logger="test_ctx"):
            logger.info("event")
        record = json.loads(caplog.records[0].message)
        assert record["node_id"] == "n1"

    def test_clear_context(self, caplog):
        logger = StructuredLogger("test_clr", level=logging.DEBUG)
        logger.set_context(x="y")
        logger.clear_context()
        with caplog.at_level(logging.INFO, logger="test_clr"):
            logger.info("clean")
        record = json.loads(caplog.records[0].message)
        assert "x" not in record


class TestTimer:
    def test_timer_records_duration(self):
        m = MetricsCollector()
        with Timer(m, "op_duration") as t:
            total = sum(range(1000))
        assert t.elapsed > 0
        stats = m.histogram("op_duration")
        assert stats["count"] == 1
        assert stats["sum"] > 0
