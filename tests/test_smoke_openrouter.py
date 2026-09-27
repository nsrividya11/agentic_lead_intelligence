"""
SMOKE TEST — hits the real OpenRouter API, no mocks.

This is the one test that can actually fail because of the outside
world (rate limits, a retired free model, a bad API key) rather than
a bug in our code. Run it right before recording the demo/submitting,
not on every code change.

Checks:
  1. The API key loads and authenticates at all.
  2. At least one free model in FREE_MODELS is currently reachable
     and returns a real completion for the exact payload shape the
     agent uses in production.
  3. The full call_llm() fallback path works against the live API
     (not just mocked responses).
  4. The end-to-end agent.score_step() produces a parseable score
     against a real LLM call, on a realistic enriched-lead payload.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from tools.llm import OPENROUTER_API_KEY, OPENROUTER_URL, FREE_MODELS, call_llm


def smoke_test_api_key_present():
    assert OPENROUTER_API_KEY, "OPENROUTER_API_KEY is empty — check .env"
    assert OPENROUTER_API_KEY.startswith("sk-or-"), \
        "OPENROUTER_API_KEY doesn't look like a valid OpenRouter key"
    print("[SMOKE] API key present and shaped correctly.")


def smoke_test_at_least_one_free_model_reachable():
    """
    Hits every model in FREE_MODELS once with a trivial payload and
    reports which ones are alive right now. Passes if >=1 responds
    with a real completion — this is exactly what call_llm() needs.
    """
    payload_template = {
        "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
        "temperature": 0.0,
    }

    alive = []
    dead = []

    for model in FREE_MODELS:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": model, **payload_template},
                timeout=30,
            )
            if resp.status_code == 200 and resp.json().get("choices"):
                alive.append(model)
            else:
                dead.append((model, resp.status_code, resp.text[:120]))
        except requests.RequestException as exc:
            dead.append((model, "EXC", str(exc)[:120]))

    print(f"[SMOKE] {len(alive)}/{len(FREE_MODELS)} free models reachable right now.")
    for m in alive:
        print(f"        OK   {m}")
    for m, code, msg in dead:
        print(f"        FAIL {m} -> {code}: {msg}")

    assert len(alive) >= 1, (
        "No free models are currently reachable — all rate-limited/retired. "
        "Refresh FREE_MODELS from GET https://openrouter.ai/api/v1/models"
    )


def smoke_test_call_llm_end_to_end_with_real_payload():
    """
    Exercises call_llm() exactly as agent.respond_step() calls it —
    same message shape, real network call, real fallback logic.
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
    print("OPENROUTER SMOKE TEST — real network calls, real API key")
    print("=" * 70)

    smoke_test_api_key_present()
    smoke_test_at_least_one_free_model_reachable()
    smoke_test_call_llm_end_to_end_with_real_payload()
    smoke_test_full_score_step_against_real_llm()

    print("=" * 70)
    print("SMOKE TEST PASSED — OpenRouter integration is live and working.")
    print("=" * 70)
