"""Tests for MonthState: persistence, keep-rate smoothing, and set tracking."""

from __future__ import annotations

import json

from fashion_forensics.mining.miner import MonthState


class TestMonthStateSaveLoad:
    """State saves and loads correctly from JSON."""

    def test_round_trip_preserves_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202501")
        state.saved_records = 10
        state.clean_records = 7
        state.estimated_keep_rate = 0.65
        state.attempted_urls.add("https://example.com/img1.jpg")
        state.saved_hashes.add("abc123")
        state.save()

        loaded = MonthState("202501")
        assert loaded.saved_records == 10
        assert loaded.clean_records == 7
        assert loaded.estimated_keep_rate == 0.65
        assert "https://example.com/img1.jpg" in loaded.attempted_urls
        assert "abc123" in loaded.saved_hashes

    def test_fresh_state_has_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202512")
        assert state.saved_records == 0
        assert state.clean_records == 0
        assert state.estimated_keep_rate == 0.5
        assert len(state.attempted_urls) == 0
        assert len(state.saved_hashes) == 0

    def test_save_creates_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202503")
        state.save()

        # File uses YYYY-MM format
        path = tmp_path / "2025-03_state.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["month"] == "202503"
        assert "updated_at" in data


class TestKeepRateSmoothing:
    """Keep rate updates with 0.7 old + 0.3 observed, clamped to [0.15, 0.9]."""

    def test_basic_smoothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202501")
        state.estimated_keep_rate = 0.5
        state.update_keep_rate(raw_added=10, clean_added=8)
        # 0.7 * 0.5 + 0.3 * 0.8 = 0.35 + 0.24 = 0.59
        assert abs(state.estimated_keep_rate - 0.59) < 1e-9

    def test_clamp_upper_bound(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202501")
        state.estimated_keep_rate = 0.9
        state.update_keep_rate(raw_added=10, clean_added=10)
        # 0.7 * 0.9 + 0.3 * 1.0 = 0.63 + 0.30 = 0.93 -> clamped to 0.9
        assert state.estimated_keep_rate == 0.9

    def test_clamp_lower_bound(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202501")
        state.estimated_keep_rate = 0.15
        state.update_keep_rate(raw_added=10, clean_added=0)
        # 0.7 * 0.15 + 0.3 * 0.0 = 0.105 -> clamped to 0.15
        assert state.estimated_keep_rate == 0.15

    def test_no_update_when_zero_raw(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202501")
        state.estimated_keep_rate = 0.5
        state.update_keep_rate(raw_added=0, clean_added=0)
        assert state.estimated_keep_rate == 0.5


class TestSetPersistence:
    """Attempted URLs and saved hashes persist across save/load."""

    def test_multiple_urls_and_hashes_persist(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202506")
        urls = {f"https://example.com/img{i}.jpg" for i in range(5)}
        hashes = {f"hash{i}" for i in range(3)}
        state.attempted_urls = urls
        state.saved_hashes = hashes
        state.save()

        loaded = MonthState("202506")
        assert loaded.attempted_urls == urls
        assert loaded.saved_hashes == hashes

    def test_used_snapshots_persist(self, tmp_path, monkeypatch):
        monkeypatch.setattr("fashion_forensics.mining.miner.STATE_DIR", tmp_path)

        state = MonthState("202507")
        state.used_snapshots = {
            "https://web.archive.org/web/snap1",
            "https://web.archive.org/web/snap2",
        }
        state.save()

        loaded = MonthState("202507")
        assert "https://web.archive.org/web/snap1" in loaded.used_snapshots
        assert "https://web.archive.org/web/snap2" in loaded.used_snapshots
        assert len(loaded.used_snapshots) == 2
