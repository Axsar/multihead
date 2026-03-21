"""Tests for resource monitoring and sparklines."""

import time



from multihead.resource_monitor import (
    ResourceSnapshot,
    ResourceTracker,
    sparkline,
)


class TestResourceSnapshot:
    def test_defaults(self):
        snap = ResourceSnapshot()
        assert snap.gpu_vram_used_mb == 0
        assert snap.gpu_vram_pct == 0.0
        assert snap.ram_pct == 0.0
        assert snap.disk_pct == 0.0

    def test_gpu_vram_pct(self):
        snap = ResourceSnapshot(gpu_vram_used_mb=6000, gpu_vram_total_mb=24000)
        assert snap.gpu_vram_pct == 25.0

    def test_ram_pct(self):
        snap = ResourceSnapshot(ram_used_mb=8000, ram_total_mb=32000)
        assert snap.ram_pct == 25.0

    def test_disk_pct(self):
        snap = ResourceSnapshot(disk_free_mb=200000, disk_total_mb=500000)
        assert snap.disk_pct == 60.0

    def test_to_dict(self):
        snap = ResourceSnapshot(
            timestamp=1000.0,
            gpu_vram_used_mb=6000,
            gpu_vram_total_mb=24000,
            ram_used_mb=8000,
            ram_total_mb=32000,
        )
        d = snap.to_dict()
        assert d["gpu_vram_used_mb"] == 6000
        assert d["gpu_vram_pct"] == 25.0
        assert d["ram_pct"] == 25.0
        assert d["timestamp"] == 1000.0

    def test_zero_total_no_division_error(self):
        snap = ResourceSnapshot(gpu_vram_total_mb=0, ram_total_mb=0, disk_total_mb=0)
        assert snap.gpu_vram_pct == 0.0
        assert snap.ram_pct == 0.0
        assert snap.disk_pct == 0.0


class TestResourceTracker:
    def test_init(self):
        tracker = ResourceTracker(history_size=10)
        assert tracker.latest is None
        assert tracker.history == []

    def test_sample(self):
        tracker = ResourceTracker(history_size=10)
        snap = tracker.sample()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.timestamp > 0
        assert tracker.latest is snap
        assert len(tracker.history) == 1

    def test_history_size_limit(self):
        tracker = ResourceTracker(history_size=3)
        for _ in range(5):
            tracker.sample()
        assert len(tracker.history) == 3

    def test_series(self):
        tracker = ResourceTracker(history_size=10)
        # Manually add snapshots
        for i in range(5):
            snap = ResourceSnapshot(gpu_vram_used_mb=i * 1000)
            tracker._history.append(snap)
        series = tracker.series("gpu_vram_used_mb")
        assert series == [0, 1000, 2000, 3000, 4000]

    def test_series_empty(self):
        tracker = ResourceTracker()
        assert tracker.series("gpu_vram_used_mb") == []

    def test_background_start_stop(self):
        tracker = ResourceTracker(history_size=5)
        tracker.start_background(interval=0.1)
        assert tracker._running
        time.sleep(0.35)
        tracker.stop()
        assert not tracker._running
        # Should have collected some samples
        assert len(tracker.history) >= 2

    def test_start_background_idempotent(self):
        tracker = ResourceTracker()
        tracker.start_background(interval=60.0)
        tracker.start_background(interval=60.0)  # second call is no-op
        assert tracker._running
        tracker.stop()


class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_single_value(self):
        result = sparkline([5.0])
        assert len(result) == 1

    def test_flat_values(self):
        result = sparkline([10, 10, 10, 10])
        assert len(result) == 4
        # All same character (middle block)
        assert len(set(result)) == 1

    def test_ascending(self):
        result = sparkline([0, 1, 2, 3, 4, 5, 6, 7])
        assert len(result) == 8
        # First char should be lowest block, last should be highest
        assert result[0] == "▁"
        assert result[-1] == "█"

    def test_descending(self):
        result = sparkline([7, 6, 5, 4, 3, 2, 1, 0])
        assert result[0] == "█"
        assert result[-1] == "▁"

    def test_width_truncation(self):
        values = list(range(50))
        result = sparkline(values, width=10)
        assert len(result) == 10  # only last 10 values

    def test_mixed_values(self):
        result = sparkline([1, 5, 3, 8, 2])
        assert len(result) == 5
        # 8 is max, should be highest block
        assert result[3] == "█"
        # 1 is min, should be lowest block
        assert result[0] == "▁"
