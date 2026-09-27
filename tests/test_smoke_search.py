"""
SMOKE TEST — hits the real Tavily search API, no mocks.

Run this before recording the demo / submitting, alongside
test_smoke_openrouter.py, to confirm both real data sources the agent
depends on are actually reachable right now.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.search import web_search, TAVILY_API_KEY


def smoke_test_api_key_present():
    assert TAVILY_API_KEY, "TAVILY_API_KEY is empty — check .env"
    assert TAVILY_API_KEY.startswith("tvly-"), "TAVILY_API_KEY doesn't look like a Tavily key"
    print("[SMOKE] Tavily API key present and shaped correctly.")


def smoke_test_real_search_returns_relevant_results():
    results = web_search("Indian ed-tech companies hiring AI engineers", max_results=5)
    assert len(results) > 0, "Tavily returned zero results for a reasonable query"
    print(f"[SMOKE] Got {len(results)} real search results:")
    for r in results:
        print(f"        - {r['title']} ({r['url']})")
        assert r["title"]
        assert r["url"]


def smoke_test_full_find_and_enrich_against_real_data():
    """
    True end-to-end: real search finds candidates, real second search
    enriches one of them — the exact path agent.find_step() and
    agent.enrich_step() take, without mocking the LLM extraction step
    (this only checks the search layer; full pipeline smoke lives in
    the manual run of `python agent.py "..."` before demo recording).
    """
    find_results = web_search("Indian ed-tech companies", max_results=3)
    assert len(find_results) > 0

    first_candidate_name = find_results[0]["title"]
    enrich_results = web_search(f'"{first_candidate_name}" employees OR hiring', max_results=3)
    print(f"[SMOKE] Enrichment search for {first_candidate_name!r} returned "
          f"{len(enrich_results)} results.")
    # Not asserting >0 here — a specific enrichment query can legitimately
    # return nothing; that's exactly the case enrich_step() must handle,
    # and is covered by the mocked unit test, not this smoke test.


if __name__ == "__main__":
    print("=" * 70)
    print("TAVILY SEARCH SMOKE TEST — real network calls, real API key")
    print("=" * 70)

    smoke_test_api_key_present()
    smoke_test_real_search_returns_relevant_results()
    smoke_test_full_find_and_enrich_against_real_data()

    print("=" * 70)
    print("SMOKE TEST PASSED — Tavily search integration is live and working.")
    print("=" * 70)
