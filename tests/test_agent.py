"""
Unit tests for agent.py — the understand/plan/find/enrich/score
orchestration logic. All LLM and web-search calls are mocked so these
run fast, free, and deterministically. Real network calls are covered
separately in tests/test_smoke_openrouter.py and tests/test_smoke_search.py.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import agent


def test_understand_step_parses_goal_into_spec():
    fake_llm_output = """{
        "search_queries": ["list of Indian ed-tech companies", "Indian ed-tech startups hiring AI engineers"],
        "target_count": 5,
        "entity_type": "company",
        "criteria": ["100+ employees", "hiring AI engineers"],
        "scoring_weights": {"100+ employees": 40, "hiring AI engineers": 60}
    }"""
    with patch("agent.call_llm", return_value=fake_llm_output):
        spec = agent.understand_step("Find 5 Indian ed-tech companies hiring AI engineers")

    assert spec["target_count"] == 5
    assert spec["entity_type"] == "company"
    assert "100+ employees" in spec["criteria"]
    assert len(spec["search_queries"]) == 2


def test_understand_step_defaults_target_count_when_missing():
    fake_llm_output = '{"search_queries": ["test query"], "criteria": [], "scoring_weights": {}}'
    with patch("agent.call_llm", return_value=fake_llm_output):
        spec = agent.understand_step("find some companies")

    assert spec["target_count"] == 5  # default


def test_understand_step_falls_back_when_search_queries_missing():
    """If the model returns the old singular search_query shape (or omits
    queries entirely), understand_step must still produce a usable spec."""
    fake_llm_output = '{"search_query": "test query", "criteria": [], "scoring_weights": {}}'
    with patch("agent.call_llm", return_value=fake_llm_output):
        spec = agent.understand_step("find some companies")

    assert spec["search_queries"] == ["test query"]


def test_understand_step_raises_on_unparseable_output():
    with patch("agent.call_llm", return_value="I don't know what you mean by that."):
        try:
            agent.understand_step("find companies")
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "Could not parse task spec" in str(exc)


def test_plan_step_reflects_the_specific_goal():
    spec = {
        "search_queries": ["colleges in Hyderabad", "list of engineering colleges Hyderabad"],
        "target_count": 3,
        "entity_type": "college",
        "criteria": ["located in Hyderabad"],
    }
    plan = agent.plan_step(spec)
    assert "3" in plan["goal"]
    assert "college" in plan["goal"]
    assert any("colleges in Hyderabad" in step for step in plan["steps"])


def test_find_step_merges_and_dedupes_across_query_variants():
    """
    find_step runs every query in search_queries and merges results,
    deduping by URL — a candidate surfaced by two different query
    variants should only appear once.
    """
    def fake_web_search(query, max_results=8):
        if query == "query A":
            return [
                {"title": "Result 1", "url": "http://shared.com", "content": "..."},
                {"title": "Result 2", "url": "http://unique-a.com", "content": "..."},
            ]
        return [
            {"title": "Result 1 dup", "url": "http://shared.com", "content": "..."},
            {"title": "Result 3", "url": "http://unique-b.com", "content": "..."},
        ]

    spec = {"search_queries": ["query A", "query B"], "target_count": 3}
    with patch("agent.web_search", side_effect=fake_web_search):
        results = agent.find_step(spec)

    urls = {r["url"] for r in results}
    assert urls == {"http://shared.com", "http://unique-a.com", "http://unique-b.com"}
    assert len(results) == 3  # shared.com counted once despite appearing in both queries


def test_find_step_continues_when_one_query_variant_fails():
    def fake_web_search(query, max_results=8):
        if query == "bad query":
            raise RuntimeError("search API down")
        return [{"title": "OK Result", "url": "http://ok.com", "content": "..."}]

    spec = {"search_queries": ["bad query", "good query"], "target_count": 3}
    with patch("agent.web_search", side_effect=fake_web_search):
        results = agent.find_step(spec)

    assert len(results) == 1
    assert results[0]["url"] == "http://ok.com"


def test_find_step_handles_search_failure_gracefully():
    spec = {"search_queries": ["test query"], "target_count": 3}
    with patch("agent.web_search", side_effect=RuntimeError("search API down")):
        results = agent.find_step(spec)
    assert results == []


def test_extract_candidate_step_returns_name_and_source():
    fake_llm_output = '{"name": "ABC University", "reason": "Matches college search"}'
    raw_result = {"title": "ABC University", "url": "http://abc.edu", "content": "..."}
    spec = {"criteria": ["located in India"], "entity_type": "college"}
    with patch("agent.call_llm", return_value=fake_llm_output):
        candidate = agent.extract_candidate_step(raw_result, spec)
    assert candidate["name"] == "ABC University"
    assert candidate["source_url"] == "http://abc.edu"


def test_extract_candidate_step_returns_none_for_non_candidate():
    fake_llm_output = '{"name": null, "reason": "not a real candidate"}'
    raw_result = {"title": "Top 10 Lists", "url": "http://listicle.com", "content": "..."}
    spec = {"criteria": ["located in India"], "entity_type": "college"}
    with patch("agent.call_llm", return_value=fake_llm_output):
        candidate = agent.extract_candidate_step(raw_result, spec)
    assert candidate is None


def test_enrich_step_returns_structured_fields():
    fake_search_results = [{"title": "ABC Corp", "url": "http://abc.com", "content": "500 employees, hiring AI engineers"}]
    fake_llm_output = """{
        "name": "ABC Corp", "industry": "Education", "size_estimate": "500+",
        "location": "Hyderabad", "hiring_signal": "Hiring AI engineers",
        "other_signal": null, "found": true
    }"""
    with patch("agent.web_search", return_value=fake_search_results), \
         patch("agent.call_llm", return_value=fake_llm_output):
        enriched = agent.enrich_step({"name": "ABC Corp", "source_url": "http://abc.com"})

    assert enriched["found"] is True
    assert enriched["industry"] == "Education"


def test_enrich_step_handles_no_search_results():
    with patch("agent.web_search", return_value=[]):
        enriched = agent.enrich_step({"name": "Unknown Co", "source_url": "http://x.com"})
    assert enriched["found"] is False


def test_enrich_step_handles_search_failure():
    with patch("agent.web_search", side_effect=RuntimeError("search failed")):
        enriched = agent.enrich_step({"name": "Unknown Co", "source_url": "http://x.com"})
    assert enriched["found"] is False


def test_score_step_parses_valid_json():
    fake_llm_output = '{"score": 87, "justification": "Strong fit."}'
    with patch("agent.call_llm", return_value=fake_llm_output):
        verdict = agent.score_step(
            {"name": "ABC Corp", "found": True, "industry": "Education"},
            {"criteria": ["hiring AI engineers"], "scoring_weights": {"hiring AI engineers": 100}},
        )
    assert verdict["score"] == 87


def test_score_step_handles_malformed_output():
    with patch("agent.call_llm", return_value="this lead seems okay, maybe 70?"):
        verdict = agent.score_step(
            {"name": "ABC Corp", "found": True}, {"criteria": [], "scoring_weights": {}}
        )
    assert verdict["score"] == 0
    assert "Could not parse" in verdict["justification"]


def test_run_pipeline_end_to_end_ranks_scored_above_unscored():
    """
    Full pipeline test with everything mocked: 2 candidates found,
    one enriches successfully and scores, one fails enrichment —
    scored candidate must rank first regardless of discovery order.
    """
    spec_output = """{
        "search_query": "test companies", "target_count": 2, "entity_type": "company",
        "criteria": ["hiring"], "scoring_weights": {"hiring": 100}
    }"""
    raw_results = [
        {"title": "Ghost Co", "url": "http://ghost.com", "content": "..."},
        {"title": "Real Co", "url": "http://real.com", "content": "..."},
    ]

    def fake_call_llm(messages, **kwargs):
        prompt = messages[0]["content"]
        if "structured task spec" in prompt:
            return spec_output
        if "SPECIFIC, NAMEABLE candidate" in prompt:
            if "Ghost Co" in prompt:
                return '{"name": "Ghost Co", "reason": "candidate"}'
            return '{"name": "Real Co", "reason": "candidate"}'
        if "Extract structured firmographic" in prompt:
            return '{"name": "Real Co", "industry": "Tech", "found": true}'
        if "Score this lead" in prompt:
            return '{"score": 85, "justification": "Good fit."}'
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

    def fake_web_search(query, max_results=8):
        if "Ghost Co" in query:
            return []  # enrichment fails for Ghost Co
        if "Real Co" in query:
            return [{"title": "Real Co info", "url": "http://real.com", "content": "hiring AI engineers"}]
        return raw_results  # initial find_step call

    with patch("agent.call_llm", side_effect=fake_call_llm), \
         patch("agent.web_search", side_effect=fake_web_search):
        results = agent.run_pipeline("find 2 test companies that are hiring")

    assert len(results) == 2
    assert results[0]["name"] == "Real Co"
    assert results[0]["score"] == 85
    assert results[1]["name"] == "Ghost Co"
    assert results[1]["score"] is None


def test_run_pipeline_returns_empty_when_no_candidates_found():
    spec_output = """{
        "search_query": "nonexistent things", "target_count": 3, "entity_type": "company",
        "criteria": [], "scoring_weights": {}
    }"""
    with patch("agent.call_llm", return_value=spec_output), \
         patch("agent.web_search", return_value=[]):
        results = agent.run_pipeline("find 3 nonexistent things")
    assert results == []


def test_run_pipeline_returns_empty_when_all_candidates_rejected_by_extraction():
    """
    Raw search results can come back non-empty (e.g. directory/listicle
    pages) but every one of them can fail the relevance check in
    extract_candidate_step. This must still surface as an empty result
    list, distinct from find_step finding zero raw results at all.
    """
    spec_output = """{
        "search_queries": ["test query"], "target_count": 3, "entity_type": "company",
        "criteria": ["some criterion"], "scoring_weights": {"some criterion": 100}
    }"""
    raw_results = [
        {"title": "List of top companies", "url": "http://directory.com", "content": "..."},
        {"title": "Another directory page", "url": "http://directory2.com", "content": "..."},
    ]

    def fake_call_llm(messages, **kwargs):
        prompt = messages[0]["content"]
        if "structured task spec" in prompt:
            return spec_output
        if "SPECIFIC, NAMEABLE candidate" in prompt:
            return '{"name": null, "reason": "this is a directory listing, not one candidate"}'
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

    with patch("agent.call_llm", side_effect=fake_call_llm), \
         patch("agent.web_search", return_value=raw_results):
        results = agent.run_pipeline("find 3 test companies")

    assert results == []


def test_run_pipeline_survives_one_candidate_exhausting_all_llm_models():
    """
    If the LLM is fully exhausted (all free models fail) while processing
    one candidate, that candidate should be marked unscored — the whole
    batch must not crash, since later candidates might still succeed.
    """
    spec_output = """{
        "search_query": "test companies", "target_count": 2, "entity_type": "company",
        "criteria": ["hiring"], "scoring_weights": {"hiring": 100}
    }"""
    raw_results = [
        {"title": "Broken Co", "url": "http://broken.com", "content": "..."},
        {"title": "Good Co", "url": "http://good.com", "content": "..."},
    ]

    call_count = {"n": 0}

    def fake_call_llm(messages, **kwargs):
        prompt = messages[0]["content"]
        if "structured task spec" in prompt:
            return spec_output
        if "SPECIFIC, NAMEABLE candidate" in prompt:
            call_count["n"] += 1
            if "Broken Co" in prompt:
                # Simulate total LLM exhaustion for this specific call.
                raise RuntimeError("All free models failed. Last error: quota exhausted")
            return '{"name": "Good Co", "reason": "candidate"}'
        if "Extract structured firmographic" in prompt:
            return '{"name": "Good Co", "industry": "Tech", "found": true}'
        if "Score this lead" in prompt:
            return '{"score": 70, "justification": "Decent fit."}'
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

    def fake_web_search(query, max_results=8):
        if "Good Co" in query:
            return [{"title": "Good Co info", "url": "http://good.com", "content": "hiring"}]
        return raw_results

    with patch("agent.call_llm", side_effect=fake_call_llm), \
         patch("agent.web_search", side_effect=fake_web_search):
        results = agent.run_pipeline("find 2 test companies")

    names = {r["name"] for r in results}
    assert "Good Co" in names
    good_result = next(r for r in results if r["name"] == "Good Co")
    assert good_result["score"] == 70


if __name__ == "__main__":
    test_understand_step_parses_goal_into_spec()
    test_understand_step_defaults_target_count_when_missing()
    test_understand_step_raises_on_unparseable_output()
    test_plan_step_reflects_the_specific_goal()
    test_find_step_returns_all_raw_results_uncapped()
    test_find_step_handles_search_failure_gracefully()
    test_extract_candidate_step_returns_name_and_source()
    test_extract_candidate_step_returns_none_for_non_candidate()
    test_enrich_step_returns_structured_fields()
    test_enrich_step_handles_no_search_results()
    test_enrich_step_handles_search_failure()
    test_score_step_parses_valid_json()
    test_score_step_handles_malformed_output()
    test_run_pipeline_end_to_end_ranks_scored_above_unscored()
    test_run_pipeline_returns_empty_when_no_candidates_found()
    test_run_pipeline_returns_empty_when_all_candidates_rejected_by_extraction()
    test_run_pipeline_survives_one_candidate_exhausting_all_llm_models()
    print("All agent tests passed.")
