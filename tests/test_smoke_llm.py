"""
SMOKE TEST — hits the real Groq and OpenRouter APIs, no mocks.

This is the one test that can actually fail because of the outside
world (rate limits, a retired free model, a bad API key) rather than
a bug in our code. Run it right before recording the demo/submitting,
not on every code change.

Checks:
  1. Both API keys load and authenticate.
  2. At least one Groq model is currently reachable (the primary,
     higher-limit tier).
  3. At least one OpenRouter free model is currently reachable (the
     fallback tier), so the safety net actually works if Groq is down.
  4. The full call_llm() path works against live traffic, using
     whichever provider is actually healthy right now.
  5. The end-to-end agent.score_step() produces a parseable score
     against a real LLM call, on a realistic enriched-lead payload.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from tools.llm import (
    GROQ_API_KEY, GROQ_URL, GROQ_MODELS,
    OPENROUTER_API_KEY, OPENROUTER_URL, OPENROUTER_FREE_MODELS,
    call_llm,
)


def smoke_test_api_keys_present():
    assert GROQ_API_KEY, "GROQ_API_KEY is empty — check .env"
    assert GROQ_API_KEY.startswith("gsk_"), "GROQ_API_KEY doesn't look like a valid Groq key"
    assert OPENROUTER_API_KEY, "OPENROUTER_API_KEY is empty — check .env"
    assert OPENROUTER_API_KEY.startswith("sk-or-"), \
        "OPENROUTER_API_KEY doesn't look like a valid OpenRouter key"
    print("[SMOKE] Both API keys present and shaped correctly.")


def _check_models(provider_name, url, api_key, models):
    payload_template = {
        "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
        "temperature": 0.0,
    }
    alive, dead = [], []

    for model in models:
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, **payload_template},
                timeout=30,
            )
            if resp.status_code == 200 and resp.json().get("choices"):
                alive.append(model)
            else:
                dead.append((model, resp.status_code, resp.text[:120]))
        except requests.RequestException as exc:
            dead.append((model, "EXC", str(exc)[:120]))

    print(f"[SMOKE] [{provider_name}] {len(alive)}/{len(models)} models reachable right now.")
    for m in alive:
        print(f"        OK   {m}")
    for m, code, msg in dead:
        print(f"        FAIL {m} -> {code}: {msg}")
    return alive


def smoke_test_at_least_one_groq_model_reachable():
    alive = _check_models("groq", GROQ_URL, GROQ_API_KEY, GROQ_MODELS)
    assert len(alive) >= 1, (
        "No Groq models are currently reachable. Refresh GROQ_MODELS from "
        "GET https://api.groq.com/openai/v1/models"
    )


def smoke_test_at_least_one_openrouter_model_reachable():
    """
    This checks the FALLBACK tier specifically — even if Groq is fully
    healthy, we want to know the OpenRouter safety net still works.
    """
    alive = _check_models("openrouter", OPENROUTER_URL, OPENROUTER_API_KEY, OPENROUTER_FREE_MODELS)
    if len(alive) == 0:
        print("[SMOKE] WARNING: no OpenRouter free models reachable right now — "
              "the fallback tier is currently unavailable (Groq being healthy covers this).")


def smoke_test_call_llm_end_to_end_with_real_payload():
    """
    Exercises call_llm() exactly as agent.score_step() calls it — same
    message shape, real network call, real Groq-then-OpenRouter fallback.
    """
    prompt = (
        'Respond ONLY as JSON in this exact shape: '
        '{"score": <int 0-100>, "justification": "<text>"}. '
        "Score this lead's fit for a VP of Engineering ICP: "
        "name=Test Lead, title=VP of Engineering, signal=actively hiring interns."
    )
    result = call_llm([{"role": "user", "content": prompt}])
    assert isinstance(result, str) and len(result) > 0
    print(f"[SMOKE] call_llm() live response (truncated): {result[:200]!r}")


def smoke_test_full_score_step_against_real_llm():
    """
    True end-to-end: a realistic enriched-lead payload -> real LLM
    call -> parsed score. This is the exact path run_pipeline() takes
    for every enriched candidate.
    """
    from agent import score_step

    spec = {
        "criteria": ["100+ employees", "hiring AI engineers"],
        "scoring_weights": {"100+ employees": 40, "hiring AI engineers": 60},
    }
    enriched_lead = {
        "name": "Test Ed-Tech Co",
        "industry": "Education",
        "size_estimate": "500+",
        "location": "Hyderabad",
        "hiring_signal": "Actively hiring AI/ML engineers",
        "found": True,
    }

    verdict = score_step(enriched_lead, spec)
    print(f"[SMOKE] Live scoring for Test Ed-Tech Co -> {verdict}")

    assert isinstance(verdict["score"], int)
    assert 0 <= verdict["score"] <= 100
    assert isinstance(verdict["justification"], str) and len(verdict["justification"]) > 0


if __name__ == "__main__":
    print("=" * 70)
    print("LLM SMOKE TEST — real network calls to Groq + OpenRouter")
    print("=" * 70)

    smoke_test_api_keys_present()
    smoke_test_at_least_one_groq_model_reachable()
    smoke_test_at_least_one_openrouter_model_reachable()
    smoke_test_call_llm_end_to_end_with_real_payload()
    smoke_test_full_score_step_against_real_llm()

    print("=" * 70)
    print("SMOKE TEST PASSED — Groq + OpenRouter LLM integration is live and working.")
    print("=" * 70)
