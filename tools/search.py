"""
Real web search tool backed by Tavily's free-tier API.

This is the agent's "Finder" and "Enrichment" tool at once: the same
search function is used to (a) discover candidate leads for a
free-text goal, and (b) look up more detail on a specific candidate.
No mock data, no hardcoded companies/leads — every result comes from
a live search.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
TAVILY_URL = "https://api.tavily.com/search"


def web_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Run a live web search. Returns a list of
    {title, url, content} dicts — 'content' is Tavily's extracted
    snippet/summary for that page, which the LLM can reason over.

    Raises RuntimeError on failure so the caller's observe step can
    decide how to handle it (this tool intentionally does not swallow
    errors — the agent needs to know a search failed).
    """
    try:
        response = requests.post(
            TAVILY_URL,
            headers={"Content-Type": "application/json"},
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Tavily search failed for query {query!r}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Tavily search returned HTTP {response.status_code} for query {query!r}: "
            f"{response.text[:300]}"
        )

    data = response.json()
    results = data.get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]
