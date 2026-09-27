"""
Unit tests for agent.py — the plan/act/observe/respond orchestration
logic. LLM calls are mocked so these run fast, free, and deterministically.
"""

import sys
import os
import json
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent


def test_plan_step_returns_goal_and_steps():
    plan = agent.plan_step("someone@example.com")
    assert "goal" in plan
    assert "someone@example.com" in plan["goal"]
    assert len(plan["steps"]) == 4


def test_act_step_calls_enrichment_tool():
    result = agent.act_step("aditi.rao@brightwave.io")
    assert result["found"] is True
    assert result["company"] == "Brightwave Analytics"


def test_observe_step_true_for_found_lead():
    found_lead = {"found": True, "company": "Acme", "title": "CEO"}
    assert agent.observe_step(found_lead) is True


def test_observe_step_false_for_missing_lead():
    missing_lead = {"found": False, "email": "ghost@nowhere.example"}
    assert agent.observe_step(missing_lead) is False


def test_respond_step_parses_valid_json_from_llm():
    fake_llm_output = '{"score": 87, "justification": "Strong fit."}'
    with patch("agent.call_llm", return_value=fake_llm_output):
        verdict = agent.respond_step(
            {"name": "Test", "company": "Acme", "title": "CEO", "raw_signal": "hiring"},
            {"target_titles": ["CEO"]},
        )
        assert verdict["score"] == 87
        assert verdict["justification"] == "Strong fit."


def test_respond_step_parses_markdown_fenced_json():
    fake_llm_output = '```json\n{"score": 60, "justification": "Decent fit."}\n```'
    with patch("agent.call_llm", return_value=fake_llm_output):
        verdict = agent.respond_step(
            {"name": "Test", "company": "Acme", "title": "CEO", "raw_signal": "hiring"},
            {"target_titles": ["CEO"]},
        )
        assert verdict["score"] == 60


def test_respond_step_handles_malformed_llm_output_gracefully():
    fake_llm_output = "I think this lead is pretty good, score around 70 maybe?"
    with patch("agent.call_llm", return_value=fake_llm_output):
        verdict = agent.respond_step(
            {"name": "Test", "company": "Acme", "title": "CEO", "raw_signal": "hiring"},
            {"target_titles": ["CEO"]},
        )
        # Must not raise — falls back to a safe default with the raw text surfaced.
        assert verdict["score"] == 0
        assert "Could not parse" in verdict["justification"]


def test_run_pipeline_ranks_scored_leads_above_unscored(monkeypatch):
    """
    Full pipeline test: 2 real seed leads (one found, one not-found via
    a monkeypatched lookup) should produce one scored + one unscored
    result, with the scored one ranked first regardless of list order.
    """

    fake_leads = [
        {"name": "Ghost Lead", "email": "ghost@nowhere.example", "company": "?", "title": "?"},
        {"name": "Aditi Rao", "email": "aditi.rao@brightwave.io", "company": "Brightwave", "title": "VP"},
    ]

    def fake_load_json(path):
        if "seed_leads" in path:
            return fake_leads
        return {"target_titles": ["VP of Engineering"]}

    fake_llm_output = '{"score": 91, "justification": "Great fit."}'

    with patch("agent.load_json", side_effect=fake_load_json), \
         patch("agent.call_llm", return_value=fake_llm_output):
        results = agent.run_pipeline()

    assert len(results) == 2
    # Scored lead ranked first even though it was second in the input list.
    assert results[0]["email"] == "aditi.rao@brightwave.io"
    assert results[0]["score"] == 91
    # Unscored lead present but placed after scored ones.
    assert results[1]["email"] == "ghost@nowhere.example"
    assert results[1]["score"] is None


if __name__ == "__main__":
    test_plan_step_returns_goal_and_steps()
    test_act_step_calls_enrichment_tool()
    test_observe_step_true_for_found_lead()
    test_observe_step_false_for_missing_lead()
    test_respond_step_parses_valid_json_from_llm()
    test_respond_step_parses_markdown_fenced_json()
    test_respond_step_handles_malformed_llm_output_gracefully()

    class _Ctx:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    test_run_pipeline_ranks_scored_leads_above_unscored(_Ctx())
    print("All agent tests passed.")
