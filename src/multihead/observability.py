"""Observability: metrics collection and structured logging."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Lightweight metrics collector with counter, gauge, and histogram support."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._labels: dict[str, dict[str, str]] = {}

    # -- Counters --

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        self._counters[key] += value
        if labels:
            self._labels[key] = labels

    def counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        return self._counters[self._key(name, labels)]

    # -- Gauges --

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        return self._gauges.get(self._key(name, labels), 0.0)

    # -- Histograms --

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        self._histograms[key].append(value)
        if labels:
            self._labels[key] = labels

    def histogram(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        key = self._key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    # -- Export --

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {"counters": {}, "gauges": {}, "histograms": {}}
        for key, val in self._counters.items():
            result["counters"][key] = {"value": val, **self._get_labels(key)}
        for key, val in self._gauges.items():
            result["gauges"][key] = {"value": val, **self._get_labels(key)}
        for key in self._histograms:
            result["histograms"][key] = {**self.histogram(key), **self._get_labels(key)}
        return result

    def to_prometheus(self) -> str:
        lines: list[str] = []
        for key, val in sorted(self._counters.items()):
            prom_labels = self._prom_labels(key)
            lines.append(f"# TYPE {self._base_name(key)} counter")
            lines.append(f"{self._base_name(key)}{prom_labels} {val}")
        for key, val in sorted(self._gauges.items()):
            prom_labels = self._prom_labels(key)
            lines.append(f"# TYPE {self._base_name(key)} gauge")
            lines.append(f"{self._base_name(key)}{prom_labels} {val}")
        for key in sorted(self._histograms):
            stats = self.histogram(key)
            prom_labels = self._prom_labels(key)
            base = self._base_name(key)
            lines.append(f"# TYPE {base} summary")
            lines.append(f"{base}_count{prom_labels} {stats['count']}")
            lines.append(f"{base}_sum{prom_labels} {stats['sum']}")
        return "\n".join(lines) + "\n" if lines else ""

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._labels.clear()

    # -- Internal --

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _get_labels(self, key: str) -> dict[str, Any]:
        labels = self._labels.get(key)
        return {"labels": labels} if labels else {}

    @staticmethod
    def _base_name(key: str) -> str:
        return key.split("{")[0] if "{" in key else key

    def _prom_labels(self, key: str) -> str:
        if "{" in key:
            return key[key.index("{"):]
        return ""


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """JSON-structured logger wrapper."""

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._context: dict[str, Any] = {}

    def set_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)

    def clear_context(self) -> None:
        self._context.clear()

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        record = {
            "ts": time.time(),
            "level": logging.getLevelName(level),
            "msg": msg,
            **self._context,
            **kwargs,
        }
        self._logger.log(level, json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Context manager that measures elapsed time and records to metrics."""

    def __init__(self, metrics: MetricsCollector, name: str, labels: dict[str, str] | None = None) -> None:
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        self.metrics.observe(self.name, self.elapsed, self.labels)
