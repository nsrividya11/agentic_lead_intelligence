"""
Lead Finder + Enrichment + Scoring Agent
=========================================

Prototype for a Techvruk-style opportunity-matching engine: given a raw
lead, the agent plans out how to evaluate it, enriches it with
firmographic/signal data, scores it against Techvruk's ICP, and returns
a ranked, reasoned verdict.

Workflow (visible at every step, printed to stdout for the demo):

    PLAN     -> decide what info is needed to score this lead
    ACT      -> call the enrichment tool to fetch that info
    OBSERVE  -> check whether the tool call succeeded; handle failure
    RESPOND  -> ask the LLM to score + justify against the ICP

This loop repeats per lead, then the agent ranks all leads and prints
a final summary — showing planning, tool use, state, and error handling
in one run.
"""

import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from tools.enrichment import search_company_info
from tools.llm import call_llm

_ICP_PATH = os.path.join(os.path.dirname(__file__), "data", "icp.json")
_LEADS_PATH = os.path.join(os.path.dirname(__file__), "data", "seed_leads.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plan_step(lead_email: str) -> dict:
    """PLAN: decide the steps needed to evaluate this lead."""
    plan = {
        "goal": f"Score lead {lead_email} against Techvruk's ICP",
        "steps": [
            "1. Enrich lead with firmographic + signal data",
            "2. Verify enrichment succeeded (observe)",
            "3. Score against ICP using LLM reasoning",
            "4. Return structured verdict with justification",
        ],
    }
    print(f"\n[PLAN] {plan['goal']}")
    for step in plan["steps"]:
        print(f"       {step}")
    return plan


def act_step(lead_email: str) -> dict:
    """ACT: call the enrichment tool."""
    print(f"[ACT]  Calling enrichment tool for {lead_email} ...")
    result = search_company_info(lead_email)
    return result


def observe_step(enrichment_result: dict) -> bool:
    """OBSERVE: check tool result; this is the error-handling seam."""
    if not enrichment_result.get("found"):
        print(f"[OBSERVE] No data found for {enrichment_result.get('email')} — "
              f"skipping scoring, marking as low-priority/unscored.")
        return False
    print(f"[OBSERVE] Enrichment succeeded: {enrichment_result['company']} "
          f"/ {enrichment_result['title']}")
    return True


def respond_step(enrichment_result: dict, icp: dict) -> dict:
    """RESPOND: ask the LLM to reason about fit and produce a score."""
    prompt = f"""You are a lead-scoring assistant for Techvruk, an opportunity-matching platform.

ICP (Ideal Customer Profile):
{json.dumps(icp, indent=2)}

Lead to evaluate:
{json.dumps(enrichment_result, indent=2)}

Score this lead's fit against the ICP from 0-100, and give a 1-2 sentence justification.
Respond ONLY as JSON in this exact shape:
{{"score": <int 0-100>, "justification": "<text>"}}
"""
    print("[RESPOND] Asking LLM to score lead against ICP ...")
    raw = call_llm([{"role": "user", "content": prompt}])

    try:
        cleaned = raw.strip().strip("`").removeprefix("json").strip()
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        parsed = {"score": 0, "justification": f"Could not parse LLM output: {raw[:200]}"}

    return parsed


def run_pipeline():
    icp = load_json(_ICP_PATH)
    leads = load_json(_LEADS_PATH)

    results = []

    for lead in leads:
        email = lead["email"]
        print("=" * 70)
        print(f"Processing lead: {lead['name']} <{email}>")

        plan_step(email)
        enrichment_result = act_step(email)
        ok = observe_step(enrichment_result)

        if not ok:
            results.append({
                "name": lead["name"],
                "email": email,
                "score": None,
                "justification": "Skipped — no enrichment data found.",
            })
            continue

        verdict = respond_step(enrichment_result, icp)
        results.append({
            "name": enrichment_result["name"],
            "email": email,
            "score": verdict.get("score"),
            "justification": verdict.get("justification"),
        })

    print("\n" + "=" * 70)
    print("FINAL RANKED RESULTS")
    print("=" * 70)

    scored = [r for r in results if r["score"] is not None]
    unscored = [r for r in results if r["score"] is None]
    scored.sort(key=lambda r: r["score"], reverse=True)

    for rank, r in enumerate(scored, start=1):
        print(f"{rank}. {r['name']} ({r['email']}) — score: {r['score']}")
        print(f"   {r['justification']}")

    for r in unscored:
        print(f"-  {r['name']} ({r['email']}) — unscored: {r['justification']}")

    return scored + unscored


if __name__ == "__main__":
    run_pipeline()
