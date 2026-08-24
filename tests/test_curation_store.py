"""Tests for the curation review queue: staging, dedup, approve/reject."""

from __future__ import annotations

import json

import pytest

from fashion_forensics.curation import store


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    labeled = tmp_path / "labeled"
    labeled.mkdir()
    queue_path = tmp_path / "queue.json"
    attributions_path = tmp_path / "attributions.jsonl"

    monkeypatch.setattr(store, "STAGING_DIR", staging)
    monkeypatch.setattr(store, "QUEUE_PATH", queue_path)
    monkeypatch.setattr(store, "LABELED_DIR", labeled)
    monkeypatch.setattr(store, "ATTRIBUTIONS_PATH", attributions_path)

    return {
        "staging": staging,
        "labeled": labeled,
        "queue_path": queue_path,
        "attributions_path": attributions_path,
    }


def _add(sandbox, content=b"fake-bytes", **overrides):
    kwargs = {
        "source_name": "Example Trend Blog",
        "source_url": "https://example.com/trends/quiet-luxury",
        "image_url": "https://example.com/images/1.jpg",
        "alt_text": "beige trench coat",
        "trend_hint": "quiet_luxury",
    }
    kwargs.update(overrides)
    return store.add_candidate(content, ".jpg", **kwargs)


class TestAddCandidate:
    def test_stages_file_and_queues_pending_record(self, sandbox):
        record = _add(sandbox)
        assert record is not None
        assert record["status"] == "pending"
        assert (sandbox["staging"] / record["filename"]).exists()
        assert store.get_candidate(record["id"]) == record

    def test_duplicate_image_bytes_are_skipped(self, sandbox):
        first = _add(sandbox, content=b"identical")
        again = _add(sandbox, content=b"identical", source_name="Different Site")
        assert first is not None
        assert again is None
        # Only one file was ever staged
        assert len(list(sandbox["staging"].iterdir())) == 1


class TestApprove:
    def test_moves_file_into_labeled_trend_dir_and_logs_attribution(self, sandbox):
        record = _add(sandbox)
        approved = store.approve(record["id"], "mob_wife", reviewer="deb")

        assert approved["status"] == "approved"
        assert approved["decided_trend"] == "mob_wife"
        dest = sandbox["labeled"] / "mob_wife" / record["filename"]
        assert dest.exists()
        assert not (sandbox["staging"] / record["filename"]).exists()

        lines = sandbox["attributions_path"].read_text().strip().splitlines()
        assert len(lines) == 1
        logged = json.loads(lines[0])
        assert logged["trend"] == "mob_wife"
        assert logged["source_url"] == "https://example.com/trends/quiet-luxury"
        assert logged["reviewer"] == "deb"

    def test_cannot_approve_twice(self, sandbox):
        record = _add(sandbox)
        store.approve(record["id"], "quiet_luxury")
        with pytest.raises(ValueError):
            store.approve(record["id"], "quiet_luxury")

    def test_unknown_candidate_raises_key_error(self, sandbox):
        with pytest.raises(KeyError):
            store.approve("does-not-exist", "quiet_luxury")


class TestReject:
    def test_deletes_staged_file_and_marks_rejected(self, sandbox):
        record = _add(sandbox)
        rejected = store.reject(record["id"], reason="not fashion")

        assert rejected["status"] == "rejected"
        assert rejected["reject_reason"] == "not fashion"
        assert not (sandbox["staging"] / record["filename"]).exists()

    def test_cannot_reject_twice(self, sandbox):
        record = _add(sandbox)
        store.reject(record["id"])
        with pytest.raises(ValueError):
            store.reject(record["id"])


class TestListAndStats:
    def test_list_candidates_filters_by_status_and_trend_hint(self, sandbox):
        pending = _add(sandbox, content=b"a", trend_hint="quiet_luxury")
        other_hint = _add(sandbox, content=b"b", trend_hint="mob_wife")
        approved = _add(sandbox, content=b"c", trend_hint="quiet_luxury")
        store.approve(approved["id"], "quiet_luxury")

        pending_records = store.list_candidates(status="pending")
        assert {r["id"] for r in pending_records} == {pending["id"], other_hint["id"]}

        filtered = store.list_candidates(status="pending", trend_hint="quiet_luxury")
        assert [r["id"] for r in filtered] == [pending["id"]]

    def test_stats_counts_by_status_and_approved_trend(self, sandbox):
        r1 = _add(sandbox, content=b"1", trend_hint="quiet_luxury")
        r2 = _add(sandbox, content=b"2")
        r3 = _add(sandbox, content=b"3")
        store.approve(r1["id"], "quiet_luxury")
        store.reject(r2["id"])

        result = store.stats()
        assert result["pending"] == 1
        assert result["approved"] == 1
        assert result["rejected"] == 1
        assert result["approved_by_trend"] == {"quiet_luxury": 1}
        assert store.get_candidate(r3["id"])["status"] == "pending"


class TestImagePath:
    def test_resolves_staging_path_while_pending_and_labeled_path_once_approved(self, sandbox):
        record = _add(sandbox)
        assert store.image_path(record["id"]) == sandbox["staging"] / record["filename"]

        store.approve(record["id"], "office_siren")
        expected = sandbox["labeled"] / "office_siren" / record["filename"]
        assert store.image_path(record["id"]) == expected

    def test_missing_candidate_returns_none(self, sandbox):
        assert store.image_path("nope") is None
