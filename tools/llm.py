"""
Thin wrapper around the OpenRouter API with fallback across multiple
free-tier models. If one model is rate-limited, the client rotates to
the next one in FREE_MODELS and retries — this doubles as the agent's
"Workflow Robustness" story for the hackathon rubric.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free-tier model slugs on OpenRouter (":free" suffix = no cost).
# OpenRouter's free-model lineup changes over time — if a slug below
# 404s, fetch the current list with:
#   GET https://openrouter.ai/api/v1/models -> filter id.endswith(":free")
# and swap it in. Nothing else in the agent needs to change.
# Ordered by observed live availability (checked via
# tests/test_smoke_openrouter.py) — models that were rate-limited at
# last check are moved to the back so the happy path doesn't waste
# retry time on them. Re-run the smoke test occasionally and reorder.
FREE_MODELS = [
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
]

_RETRY_WAIT_SECONDS = 8


def _try_model(model: str, messages: list[dict], temperature: float) -> tuple[str | None, str | None]:
    """Call one model once. Returns (content, error)."""
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages, "temperature": temperature},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("error"):
                # Some providers return HTTP 200 with an error embedded in the
                # body (e.g. upstream rate limit) instead of a real HTTP status.
                err_code = data["error"].get("code")
                return None, f"{model} -> embedded error (code {err_code}): {data['error'].get('message')}"
            if data.get("choices"):
                return data["choices"][0]["message"]["content"], None
            return None, f"{model} -> HTTP 200 but no choices: {response.text[:300]}"
        return None, f"{model} -> HTTP {response.status_code}: {response.text[:200]}"
    except requests.RequestException as exc:
        return None, f"{model} -> {exc}"


def call_llm(messages: list[dict], temperature: float = 0.2) -> str:
    """
    Call free models in order. On a transient rate limit, retry that
    same model once after a short wait before moving to the next model.
    A daily-quota exhaustion ("free-models-per-day") is NOT transient —
    waiting won't help within this run, so it skips straight to the
    next model instead of wasting a retry.
    """
    last_error = None
    for model in FREE_MODELS:
        content, error = _try_model(model, messages, temperature)
        if content is not None:
            return content
        last_error = error

        is_daily_quota_exhausted = error and "free-models-per-day" in error
        is_transient_rate_limit = (
            not is_daily_quota_exhausted
            and error
            and any(code in error for code in ("429", "502", "ResourceExhausted"))
        )

        if is_transient_rate_limit:
            print(f"[OBSERVE] {model} rate-limited — retrying once after "
                  f"{_RETRY_WAIT_SECONDS}s before falling back ...")
            time.sleep(_RETRY_WAIT_SECONDS)
            content, error = _try_model(model, messages, temperature)
            if content is not None:
                return content
            last_error = error
        elif is_daily_quota_exhausted:
            print(f"[OBSERVE] {model} hit its daily free-tier quota — "
                  f"skipping retry, falling back to next free model ...")

        print(f"[OBSERVE] {model} failed, falling back to next free model ...")

    raise RuntimeError(f"All free models failed. Last error: {last_error}")
