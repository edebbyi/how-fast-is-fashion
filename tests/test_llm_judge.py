"""Tests for src/fashion_forensics/llm_judge.py - the LLM-assisted first-pass
judging helper for Discover's ground-truth review. Mocks
fashion_forensics.prompt_lab.run_prompt() throughout - no real LLM calls,
no network, no cost.
"""

from __future__ import annotations

from fashion_forensics import llm_judge


def test_group_by_trend_groups_correctly():
    trend_pred_by_id = {"a": "basics", "b": "quiet_luxury", "c": "basics"}
    grouped = llm_judge.group_by_trend(["a", "b", "c"], trend_pred_by_id)
    assert grouped == {"basics": ["a", "c"], "quiet_luxury": ["b"]}


def test_group_by_trend_drops_unknown_record_ids():
    trend_pred_by_id = {"a": "basics"}
    grouped = llm_judge.group_by_trend(["a", "missing"], trend_pred_by_id)
    assert grouped == {"basics": ["a"]}


def test_build_judge_prompt_includes_trend_definition_and_discriminating_details():
    prompt = llm_judge._build_judge_prompt("quiet_luxury")
    assert "quiet_luxury" in prompt
    assert "Quiet Luxury" in prompt
    assert "structured shoulder" in prompt  # discriminating_details, underscores replaced
    assert "{{product_name}}" in prompt  # left for run_prompt()'s own substitution


def test_build_judge_prompt_excludes_current_trend_from_other_trends_list():
    prompt = llm_judge._build_judge_prompt("quiet_luxury")
    # "Quiet Luxury / Old Money" (the name) should appear exactly once - in
    # the main prediction block, not duplicated in the "other trends" list.
    assert prompt.count("Quiet Luxury / Old Money") == 1


def test_build_judge_prompt_basics_uses_defined_by_absence_framing():
    """basics has no discriminating_details (empty rule, by design) - the
    prompt should say so instead of an empty "Look for:" line."""
    prompt = llm_judge._build_judge_prompt("basics")
    assert "Defined by absence" in prompt


def test_build_judge_prompt_basics_includes_other_trends_discriminating_signal():
    """Regression test for a real miss: a first version of this prompt gave
    only quiet_luxury's one-line description in the "other trends"
    reference list when judging a basics prediction - nothing connected a
    wool/cashmere product name to quiet_luxury's actual fine-material
    signal, and a real test call missed exactly that case. The prompt must
    surface quiet_luxury's discriminating material list, not just its
    vague high-level description, and must explicitly call out material
    text as sufficient on its own."""
    prompt = llm_judge._build_judge_prompt("basics")
    assert "cashmere" in prompt.lower()
    assert "fine material" in prompt.lower()


def test_judge_items_parses_valid_responses(monkeypatch):
    def fake_run_prompt(prompt_text, record_ids, **kwargs):
        return [
            {
                "record_id": rid,
                "parsed": {
                    "correct": False,
                    "reasoning": "too glamorous",
                    "corrected_trend": "mob_wife",
                },
            }
            for rid in record_ids
        ]

    monkeypatch.setattr(llm_judge, "run_prompt", fake_run_prompt)

    result = llm_judge.judge_items({"basics": ["2024-02_001"]})
    assert result == {
        "2024-02_001": {
            "correct": False,
            "reasoning": "too glamorous",
            "corrected_trend": "mob_wife",
        }
    }


def test_judge_items_skips_errored_records(monkeypatch):
    def fake_run_prompt(prompt_text, record_ids, **kwargs):
        return [{"record_id": record_ids[0], "error": "image missing on disk"}]

    monkeypatch.setattr(llm_judge, "run_prompt", fake_run_prompt)

    result = llm_judge.judge_items({"basics": ["2024-02_001"]})
    assert result == {}


def test_judge_items_skips_malformed_json_response(monkeypatch):
    def fake_run_prompt(prompt_text, record_ids, **kwargs):
        return [
            {"record_id": record_ids[0], "parsed": {"_parse_error": "bad json", "_raw": "not json"}}
        ]

    monkeypatch.setattr(llm_judge, "run_prompt", fake_run_prompt)

    result = llm_judge.judge_items({"basics": ["2024-02_001"]})
    assert result == {}


def test_judge_items_drops_invalid_corrected_trend(monkeypatch):
    """An LLM hallucinating a trend name outside the closed taxonomy
    shouldn't silently corrupt corrected_trend - drop to None instead."""

    def fake_run_prompt(prompt_text, record_ids, **kwargs):
        return [
            {
                "record_id": record_ids[0],
                "parsed": {"correct": False, "reasoning": "hmm", "corrected_trend": "streetwear"},
            }
        ]

    monkeypatch.setattr(llm_judge, "run_prompt", fake_run_prompt)

    result = llm_judge.judge_items({"basics": ["2024-02_001"]})
    assert result["2024-02_001"]["corrected_trend"] is None


def test_judge_items_calls_run_prompt_once_per_distinct_trend(monkeypatch):
    calls = []

    def fake_run_prompt(prompt_text, record_ids, **kwargs):
        calls.append((prompt_text, tuple(record_ids)))
        return [
            {
                "record_id": rid,
                "parsed": {"correct": True, "reasoning": "ok", "corrected_trend": None},
            }
            for rid in record_ids
        ]

    monkeypatch.setattr(llm_judge, "run_prompt", fake_run_prompt)

    llm_judge.judge_items({"basics": ["a", "b"], "quiet_luxury": ["c"]})
    assert len(calls) == 2
    prompts_used = {p for p, _ in calls}
    assert len(prompts_used) == 2  # each trend gets its own compiled prompt


def test_judge_items_skips_empty_groups(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_judge, "run_prompt", lambda *a, **k: calls.append(1) or [])

    llm_judge.judge_items({"basics": []})
    assert calls == []
