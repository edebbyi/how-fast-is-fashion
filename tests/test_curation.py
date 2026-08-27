"""Tests for src/fashion_forensics/curation.py - the trend-image curator's
shared candidate storage, dedup, and promotion logic. No network/Streamlit
needed; filesystem paths are redirected into tmp_path via monkeypatch so
these never touch the real data/02_reference_corpus/.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from fashion_forensics import curation


@pytest.fixture
def isolated_corpus(tmp_path, monkeypatch):
    """Redirect curation.py's module-level paths (including PROJECT_ROOT,
    since approve_candidate resolves local_path relative to it) into an
    isolated tmp_path tree."""
    labeled_root = tmp_path / "data" / "02_reference_corpus" / "labeled"
    raw_root = tmp_path / "data" / "02_reference_corpus" / "raw"
    labeled_root.mkdir(parents=True)
    raw_root.mkdir(parents=True)

    monkeypatch.setattr(curation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(curation, "LABELED_ROOT", labeled_root)
    monkeypatch.setattr(curation, "RAW_ROOT", raw_root)
    monkeypatch.setattr(curation, "CANDIDATES_PATH", tmp_path / "candidates.jsonl")
    monkeypatch.setattr(curation, "PHASHES_PATH", tmp_path / "phashes.json")
    monkeypatch.setattr(curation, "LAST_SYNC_PATH", tmp_path / ".last_sync")
    return {"labeled_root": labeled_root, "raw_root": raw_root, "project_root": tmp_path}


def _candidate_record(candidate_id: str, local_path: str, phash: str = "0000000000000000") -> dict:
    return {
        "candidate_id": candidate_id,
        "source_site": "testsite",
        "source_url": "https://example.com/article",
        "source_image_url": "https://example.com/img.jpg",
        "scraped_at": "2026-08-24T00:00:00+00:00",
        "suggested_trend": "quiet_luxury",
        "suggestion_basis": "curated_article",
        "phash": phash,
        "md5": "deadbeef",
        "local_path": local_path,
        "review_status": "pending",
        "reviewed_trend": None,
        "reject_reason": None,
        "promoted_filename": None,
        "promoted_at": None,
        "dup_of": None,
    }


class TestLoadSaveCandidates:
    def test_load_returns_empty_dict_when_no_file(self, isolated_corpus):
        assert curation.load_candidates() == {}

    def test_save_then_load_roundtrips(self, isolated_corpus):
        records = {
            "c1": _candidate_record("c1", "data/02_reference_corpus/raw/quiet_luxury/c1.jpg")
        }
        curation.save_candidates(records)
        assert curation.load_candidates() == records

    def test_save_sorts_by_candidate_id(self, isolated_corpus):
        records = {
            "zzz": _candidate_record("zzz", "raw/x/zzz.jpg"),
            "aaa": _candidate_record("aaa", "raw/x/aaa.jpg"),
        }
        curation.save_candidates(records)
        lines = curation.CANDIDATES_PATH.read_text().splitlines()
        ids = [json.loads(line)["candidate_id"] for line in lines]
        assert ids == ["aaa", "zzz"]


class TestDuplicateDetection:
    def test_identical_hash_is_duplicate(self, isolated_corpus):
        candidates = {"c1": _candidate_record("c1", "x", phash="a1b2c3d4e5f60718")}
        assert curation.find_duplicate("a1b2c3d4e5f60718", candidates, {}) == "c1"

    def test_hash_one_bit_off_is_still_duplicate(self, isolated_corpus):
        # 0x8 = 1000, 0x9 = 1001 -> 1 bit different, well under the threshold (8)
        candidates = {"c1": _candidate_record("c1", "x", phash="a1b2c3d4e5f60718")}
        assert curation.find_duplicate("a1b2c3d4e5f60719", candidates, {}) == "c1"

    def test_hash_far_beyond_threshold_is_not_duplicate(self, isolated_corpus):
        candidates = {"c1": _candidate_record("c1", "x", phash="0000000000000000")}
        assert curation.find_duplicate("ffffffffffffffff", candidates, {}) is None

    def test_checks_reference_phashes_when_no_candidate_match(self, isolated_corpus):
        reference_phashes = {"data/.../basic-1.jpg": "1111111111111111"}
        dup = curation.find_duplicate("1111111111111111", {}, reference_phashes)
        assert dup == "data/.../basic-1.jpg"

    def test_no_match_anywhere_returns_none(self, isolated_corpus):
        assert curation.find_duplicate("abcdef0123456789", {}, {}) is None


class TestUniqueFilename:
    def test_returns_stem_when_free(self, tmp_path):
        assert curation.unique_filename(tmp_path, "scraped_abc", ".jpg") == "scraped_abc.jpg"

    def test_appends_counter_on_collision(self, tmp_path):
        (tmp_path / "scraped_abc.jpg").write_bytes(b"x")
        assert curation.unique_filename(tmp_path, "scraped_abc", ".jpg") == "scraped_abc-2.jpg"

    def test_increments_past_multiple_collisions(self, tmp_path):
        (tmp_path / "scraped_abc.jpg").write_bytes(b"x")
        (tmp_path / "scraped_abc-2.jpg").write_bytes(b"x")
        assert curation.unique_filename(tmp_path, "scraped_abc", ".jpg") == "scraped_abc-3.jpg"


class TestApproveReject:
    def _stage_candidate(
        self, isolated_corpus, candidate_id="c1", filename="c1.jpg", trend="quiet_luxury"
    ):
        src_dir = isolated_corpus["raw_root"] / trend
        src_dir.mkdir(exist_ok=True)
        src_path = src_dir / filename
        src_path.write_bytes(b"fake image bytes")
        rel_path = str(src_path.relative_to(isolated_corpus["project_root"]))
        curation.save_candidates({candidate_id: _candidate_record(candidate_id, rel_path)})
        return src_path

    def test_approve_moves_file_with_scraped_prefix(self, isolated_corpus):
        src_path = self._stage_candidate(isolated_corpus)

        record = curation.approve_candidate("c1", "quiet_luxury")

        assert record["review_status"] == "approved"
        assert record["reviewed_trend"] == "quiet_luxury"
        assert record["promoted_filename"] == "scraped_c1.jpg"
        assert record["promoted_at"] is not None
        assert not src_path.exists()

        dest = isolated_corpus["labeled_root"] / "quiet_luxury" / "scraped_c1.jpg"
        assert dest.exists()
        assert dest.read_bytes() == b"fake image bytes"

    def test_approve_persists_status_to_disk(self, isolated_corpus):
        self._stage_candidate(isolated_corpus)
        curation.approve_candidate("c1", "quiet_luxury")
        assert curation.load_candidates()["c1"]["review_status"] == "approved"

    def test_approve_can_correct_the_trend(self, isolated_corpus):
        self._stage_candidate(isolated_corpus, trend="quiet_luxury")
        record = curation.approve_candidate("c1", "basics")  # human overrides the suggestion
        assert record["reviewed_trend"] == "basics"
        assert (isolated_corpus["labeled_root"] / "basics" / "scraped_c1.jpg").exists()

    def test_reject_keeps_file_and_sets_reason(self, isolated_corpus):
        src_path = self._stage_candidate(isolated_corpus)
        record = curation.reject_candidate("c1", "rejected", reason="off-trend")
        assert record["review_status"] == "rejected"
        assert record["reject_reason"] == "off-trend"
        assert src_path.exists()  # not deleted - kept for audit

    def test_duplicate_status_needs_no_reason(self, isolated_corpus):
        self._stage_candidate(isolated_corpus)
        record = curation.reject_candidate("c1", "duplicate")
        assert record["review_status"] == "duplicate"
        assert record["reject_reason"] is None


class TestSyncTracking:
    def test_pending_sync_count_zero_when_nothing_approved(self, isolated_corpus):
        assert curation.pending_sync_count() == 0

    def test_pending_sync_count_counts_approved_candidates(self, isolated_corpus):
        record = _candidate_record("c1", "x")
        record["review_status"] = "approved"
        record["promoted_at"] = "2026-08-24T12:00:00+00:00"
        curation.save_candidates({"c1": record})
        assert curation.pending_sync_count() == 1

    def test_pending_sync_count_excludes_non_approved(self, isolated_corpus):
        record = _candidate_record("c1", "x")
        record["review_status"] = "rejected"
        curation.save_candidates({"c1": record})
        assert curation.pending_sync_count() == 0

    def test_mark_synced_excludes_already_synced_approvals(self, isolated_corpus):
        record = _candidate_record("c1", "x")
        record["review_status"] = "approved"
        record["promoted_at"] = "2026-08-24T12:00:00+00:00"
        curation.save_candidates({"c1": record})
        assert curation.pending_sync_count() == 1

        curation.mark_synced()
        assert curation.pending_sync_count() == 0

        # a later approval should count again
        record2 = _candidate_record("c2", "y")
        record2["review_status"] = "approved"
        record2["promoted_at"] = "2099-01-01T00:00:00+00:00"
        candidates = curation.load_candidates()
        candidates["c2"] = record2
        curation.save_candidates(candidates)
        assert curation.pending_sync_count() == 1


class TestReferencePhashes:
    def test_computes_and_caches_phashes_for_labeled_images(self, isolated_corpus):
        trend_dir = isolated_corpus["labeled_root"] / "basics"
        trend_dir.mkdir()
        img_path = trend_dir / "basic-1.jpg"
        Image.new("RGB", (16, 16), color=(120, 60, 200)).save(img_path)

        phashes = curation.load_reference_phashes()
        rel = str(img_path.relative_to(isolated_corpus["project_root"]))
        assert rel in phashes
        assert curation.PHASHES_PATH.exists()

    def test_second_call_reuses_cache_without_recomputing(self, isolated_corpus, monkeypatch):
        trend_dir = isolated_corpus["labeled_root"] / "basics"
        trend_dir.mkdir()
        img_path = trend_dir / "basic-1.jpg"
        Image.new("RGB", (16, 16), color=(10, 200, 30)).save(img_path)

        first = curation.load_reference_phashes()

        calls = []
        original = curation.compute_phash
        monkeypatch.setattr(curation, "compute_phash", lambda b: calls.append(1) or original(b))
        second = curation.load_reference_phashes()

        assert second == first
        assert calls == []  # nothing new to hash, cache was reused
