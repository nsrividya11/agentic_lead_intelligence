"""
Lead Intelligence Agent — Finder + Enrichment + Scoring
=========================================================

Takes ANY free-text lead-generation goal (e.g. "Find 8 Indian ed-tech
companies with 100+ employees that are hiring AI engineers") and
autonomously:

    UNDERSTAND -> LLM turns the goal into a structured task spec
                  (search query, target count, scoring criteria) —
                  nothing about criteria or targets is hardcoded.
    PLAN       -> decide the concrete steps needed for this specific goal
    FIND       -> real web search (Tavily) to discover candidate leads
    ACT/ENRICH -> a second, targeted real web search per candidate,
                  then the LLM extracts structured fields from the
                  raw results (size, hiring signals, industry, etc.)
    OBSERVE    -> check each tool call's result; skip gracefully on failure
    SCORE      -> LLM scores each enriched candidate against the
                  criteria derived from the ORIGINAL goal
    RESPOND    -> ranked, reasoned leaderboard

Every run is driven entirely by the user's question — there is no
seed lead list and no static ICP file. Real web data only.
"""

import json
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from tools.search import web_search
from tools.llm import call_llm
from tools.json_utils import parse_llm_json


def understand_step(user_goal: str) -> dict:
    """
    UNDERSTAND: turn a free-text goal into a structured task spec.
    This replaces any hardcoded ICP/criteria file — the criteria are
    derived fresh from whatever the user actually asked.
    """
    prompt = f"""A user gave this lead-generation goal:

"{user_goal}"

Turn this into a structured task spec. Respond ONLY as JSON in this exact shape:
{{
  "search_queries": ["<query 1>", "<query 2>", "<query 3>"],
  "target_count": <int, how many leads the user wants; default to 5 if not specified>,
  "entity_type": "<what kind of entity is being searched for, e.g. 'company', 'college', 'person'>",
  "criteria": ["<short criterion 1>", "<short criterion 2>", "..."],
  "scoring_weights": {{"<criterion>": <int weight 0-100, all weights sum to 100>}}
}}

For "search_queries", give 2-4 DIFFERENT search queries designed to surface pages that
list, name, or profile SPECIFIC real entities matching the goal (e.g. "list of", "top",
"directory", "startups", a specific well-known industry publication/database name) —
NOT generic keyword-stuffed queries that mostly surface news articles or blog posts
ABOUT the topic rather than pages naming actual candidates.

The criteria and weights must be derived from what the user actually asked for — do not
invent generic criteria that aren't implied by the goal."""

    print(f"\n[UNDERSTAND] Parsing goal into a structured task spec ...")
    raw = call_llm([{"role": "user", "content": prompt}])
    spec = parse_llm_json(raw)

    if spec is None:
        raise RuntimeError(f"Could not parse task spec from LLM output: {raw[:300]}")

    spec.setdefault("target_count", 5)
    if not spec.get("search_queries"):
        # Defensive fallback if the model returns the old single-query shape.
        spec["search_queries"] = [spec["search_query"]] if spec.get("search_query") else [user_goal]

    print(f"[UNDERSTAND] search_queries={spec.get('search_queries')}, "
          f"target_count={spec.get('target_count')}, criteria={spec.get('criteria')}")
    return spec


def plan_step(spec: dict) -> dict:
    """PLAN: decide the concrete steps for THIS goal (not a fixed script)."""
    queries_str = "; ".join(spec.get("search_queries", []))
    plan = {
        "goal": f"Find, enrich, and rank {spec['target_count']} {spec.get('entity_type', 'lead')}(s)",
        "steps": [
            f"1. Search the web for candidates using {len(spec.get('search_queries', []))} "
            f"query variant(s): {queries_str}",
            "2. For each candidate, run a targeted follow-up search to enrich it",
            "3. Extract structured fields from search results (LLM)",
            f"4. Score each candidate against: {', '.join(spec.get('criteria', []))}",
            "5. Rank and return the leaderboard",
        ],
    }
    print(f"\n[PLAN] {plan['goal']}")
    for step in plan["steps"]:
        print(f"       {step}")
    return plan


def find_step(spec: dict) -> list[dict]:
    """
    FIND: real web search to discover candidate leads for this goal.
    Runs several query variants (from understand_step) rather than one,
    since a single generic query tends to surface news/blog posts ABOUT
    a topic rather than pages naming actual candidates. Results across
    queries are merged and deduped by URL.
    """
    queries = spec.get("search_queries") or [spec.get("search_query", "")]
    print(f"\n[FIND] Searching the web for candidates using {len(queries)} query variant(s) ...")

    per_query_limit = max(4, spec["target_count"])
    seen_urls = set()
    merged_results = []

    for query in queries:
        try:
            results = web_search(query, max_results=per_query_limit)
        except RuntimeError as exc:
            print(f"[OBSERVE] Search failed for query {query!r}: {exc}")
            continue

        new_count = 0
        for r in results:
            if r["url"] in seen_urls:
                continue
            seen_urls.add(r["url"])
            merged_results.append(r)
            new_count += 1
        print(f"[OBSERVE] Query {query!r} -> {len(results)} results ({new_count} new)")

    print(f"[OBSERVE] {len(merged_results)} unique raw search results across all queries")
    return merged_results


def extract_candidate_step(raw_result: dict, spec: dict) -> dict:
    """
    From one raw search result (title/url/content), ask the LLM to
    pull out a candidate name/organization IF it genuinely matches the
    user's original criteria — the first-pass structuring and relevance
    filter of a Finder result into something Enrichment can work with.
    """
    prompt = f"""A user is looking for leads matching these criteria: {json.dumps(spec.get("criteria", []))}
Entity type they want: {spec.get("entity_type", "organization")}

Does this web search result represent a SPECIFIC, NAMEABLE candidate that plausibly
matches the entity type and criteria above? Reject generic/well-known entities that
were only mentioned in passing (e.g. a huge global company appearing in an unrelated
listicle), off-topic results, list articles, ads, or anything not clearly a real
distinct candidate matching what the user asked for.

Respond ONLY as JSON:
{{"name": "<organization or person name>", "reason": "<why this matches, 1 sentence>"}}

If it does not clearly match, respond with:
{{"name": null, "reason": "<why it does not match>"}}

Title: {raw_result['title']}
URL: {raw_result['url']}
Content: {raw_result['content'][:800]}"""

    raw = call_llm([{"role": "user", "content": prompt}])
    parsed = parse_llm_json(raw)
    if parsed is None or not parsed.get("name"):
        return None
    return {"name": parsed["name"], "source_url": raw_result["url"]}


def enrich_step(candidate: dict) -> dict:
    """
    ACT (Enrichment): a second, targeted real web search on this
    specific candidate, to gather the detail Scoring needs.
    """
    name = candidate["name"]
    print(f"[ACT]  Enriching candidate: {name} ...")
    query = f'"{name}" employees OR hiring OR funding OR headquarters'
    try:
        results = web_search(query, max_results=4)
    except RuntimeError as exc:
        print(f"[OBSERVE] Enrichment search failed for {name}: {exc}")
        return {"name": name, "found": False}

    if not results:
        print(f"[OBSERVE] No enrichment data found for {name}")
        return {"name": name, "found": False}

    combined_content = "\n\n".join(r["content"][:500] for r in results)
    prompt = f"""Extract structured firmographic details about "{name}" from these search snippets.
Respond ONLY as JSON:
{{"name": "{name}", "industry": "<industry or null>", "size_estimate": "<employee count/range or null>",
  "location": "<location or null>", "hiring_signal": "<any hiring/growth signal mentioned, or null>",
  "other_signal": "<any other relevant buying signal, or null>", "found": true}}

Search snippets:
{combined_content}"""

    raw = call_llm([{"role": "user", "content": prompt}])
    parsed = parse_llm_json(raw)
    if parsed is None:
        print(f"[OBSERVE] Could not parse enrichment for {name}")
        return {"name": name, "found": False}

    parsed["found"] = True
    print(f"[OBSERVE] Enriched {name}: industry={parsed.get('industry')}, "
          f"size={parsed.get('size_estimate')}")
    return parsed


def score_step(enriched: dict, spec: dict) -> dict:
    """SCORE: LLM scores this enriched candidate against the criteria
    derived from the ORIGINAL user goal (not a static rubric)."""
    prompt = f"""Score this lead against the target criteria.

Target criteria: {json.dumps(spec.get("criteria", []))}
Scoring weights: {json.dumps(spec.get("scoring_weights", {}))}

Lead data:
{json.dumps(enriched, indent=2)}

Respond ONLY as JSON: {{"score": <int 0-100>, "justification": "<1-2 sentences>"}}"""

    print(f"[RESPOND] Scoring {enriched['name']} ...")
    raw = call_llm([{"role": "user", "content": prompt}])
    parsed = parse_llm_json(raw)
    if parsed is None:
        return {"score": 0, "justification": f"Could not parse LLM output: {raw[:200]}"}
    return parsed


def run_pipeline(user_goal: str, on_phase=None) -> list[dict]:
    """
    Run the full pipeline. `on_phase`, if given, is called at each
    major checkpoint, so a caller (e.g. a UI) can render live progress
    without parsing stdout:

      on_phase(phase, status, detail="", candidate_name=None)

    phase: "understand" | "plan" | "find" | "candidate"
    status: "start" | "done" | "skip" | "error"
    detail: short human-readable context (counts, score, reason, etc.)
    candidate_name: set only when phase == "candidate" — the candidate
      this event is about, kept separate from `detail` so a name
      containing punctuation never has to be parsed back out.
    """
    def notify(phase, status, detail="", candidate_name=None):
        if on_phase is not None:
            on_phase(phase, status, detail, candidate_name)

    notify("understand", "start")
    spec = understand_step(user_goal)
    notify("understand", "done", f"{len(spec.get('criteria', []))} criteria derived")

    notify("plan", "start")
    plan_step(spec)
    notify("plan", "done")

    notify("find", "start")
    raw_results = find_step(spec)
    notify("find", "done", f"{len(raw_results)} raw results")

    if not raw_results:
        print("\n[RESPOND] No candidates found for this goal.")
        notify("score", "done", "no candidates found")
        return []

    results = []
    seen_names = set()
    target_count = spec["target_count"]
    rejected_count = 0

    for raw_result in raw_results:
        if len(results) >= target_count:
            break

        try:
            candidate = extract_candidate_step(raw_result, spec)
        except RuntimeError as exc:
            # All free LLM models exhausted for this call — this candidate
            # can't be processed, but the rest of the batch still can be.
            print(f"[OBSERVE] Could not extract candidate from {raw_result.get('title')!r}: {exc}")
            continue

        if candidate is None:
            rejected_count += 1
            print(f"[OBSERVE] Not a specific candidate, skipped: {raw_result.get('title')!r}")
            continue

        # Different search results can surface the same organization twice.
        dedup_key = candidate["name"].strip().lower()
        if dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)

        notify("candidate", "start", candidate_name=candidate["name"])

        try:
            enriched = enrich_step(candidate)
        except RuntimeError as exc:
            print(f"[OBSERVE] Enrichment LLM call failed for {candidate['name']}: {exc}")
            notify("candidate", "error", "enrichment LLM unavailable", candidate["name"])
            results.append({
                "name": candidate["name"],
                "score": None,
                "justification": "Skipped — enrichment failed (LLM unavailable).",
            })
            continue

        if not enriched.get("found"):
            notify("candidate", "skip", "no enrichment data", candidate["name"])
            results.append({
                "name": candidate["name"],
                "score": None,
                "justification": "Skipped — no enrichment data found.",
            })
            continue

        try:
            verdict = score_step(enriched, spec)
        except RuntimeError as exc:
            print(f"[OBSERVE] Scoring LLM call failed for {enriched['name']}: {exc}")
            notify("candidate", "error", "scoring LLM unavailable", enriched["name"])
            results.append({
                "name": enriched["name"],
                "score": None,
                "justification": "Skipped — scoring failed (LLM unavailable).",
            })
            continue

        notify("candidate", "done", f"scored {verdict.get('score')}", enriched["name"])
        results.append({
            "name": enriched["name"],
            "score": verdict.get("score"),
            "justification": verdict.get("justification"),
            "details": enriched,
        })

    print("\n" + "=" * 70)
    print("FINAL RANKED RESULTS")
    print("=" * 70)

    if not results:
        print(f"No candidates survived extraction — {rejected_count} of "
              f"{len(raw_results)} raw search results were rejected as not being "
              f"a specific, nameable match (e.g. directory/listicle pages rather "
              f"than pages naming one real candidate). Try rephrasing the goal or "
              f"making it more specific.")
        return []

    scored = [r for r in results if r["score"] is not None]
    unscored = [r for r in results if r["score"] is None]
    scored.sort(key=lambda r: r["score"], reverse=True)

    for rank, r in enumerate(scored, start=1):
        print(f"{rank}. {r['name']} — score: {r['score']}")
        print(f"   {r['justification']}")

    for r in unscored:
        print(f"-  {r['name']} — unscored: {r['justification']}")

    return scored + unscored


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip()
    if not goal:
        goal = input("Enter your lead-generation goal: ").strip()

    try:
        run_pipeline(goal)
    except RuntimeError as exc:
        print(f"\n[ERROR] Could not complete the run: {exc}")
        print("All free-tier LLM models (Groq and OpenRouter) may be exhausted for today — "
              "try again later, or add a paid key for higher limits.")
        sys.exit(1)
