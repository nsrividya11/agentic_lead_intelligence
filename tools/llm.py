"""
LLM client with a two-tier free-tier provider fallback: Groq first
(much higher free daily limits), OpenRouter free models second. If a
model on either provider is rate-limited or exhausted, the client
rotates to the next one — this is the agent's "Workflow Robustness"
story for the hackathon rubric, now spanning providers as well as
models within a provider.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Groq's free tier has much higher daily limits than OpenRouter's
# free models, so it's tried first. Groq's model lineup changes over
# time — if a slug below 404s, fetch the current list with:
#   GET https://api.groq.com/openai/v1/models (Authorization: Bearer <key>)
# and swap it in.
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# Free-tier model slugs on OpenRouter (":free" suffix = no cost), used
# as fallback once Groq is exhausted. OpenRouter's free-model lineup
# changes over time — if a slug below 404s, fetch the current list with:
#   GET https://openrouter.ai/api/v1/models -> filter id.endswith(":free")
# and swap it in. Nothing else in the agent needs to change.
OPENROUTER_FREE_MODELS = [
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

# Ordered (provider, url, api_key, model) tuples — Groq models first,
# OpenRouter free models as fallback once Groq is exhausted/unavailable.
_PROVIDER_CHAIN = [("groq", GROQ_URL, GROQ_API_KEY, m) for m in GROQ_MODELS] + [
    ("openrouter", OPENROUTER_URL, OPENROUTER_API_KEY, m) for m in OPENROUTER_FREE_MODELS
]

_RETRY_WAIT_SECONDS = 8


def _try_model(url: str, api_key: str, model: str, messages: list[dict], temperature: float) -> tuple[str | None, str | None]:
    """Call one model on one provider once. Returns (content, error)."""
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
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
    Call Groq models first (higher free daily limits), then fall back
    to OpenRouter's free models. On a transient rate limit, retry that
    same model once after a short wait before moving to the next one.
    A daily-quota exhaustion ("free-models-per-day", or a Groq
    rate_limit_exceeded) is NOT transient — waiting won't help within
    this run, so it skips straight to the next model instead of
    wasting a retry.
    """
    last_error = None
    for provider, url, api_key, model in _PROVIDER_CHAIN:
        if not api_key:
            continue  # provider not configured (e.g. no GROQ_API_KEY set)

        content, error = _try_model(url, api_key, model, messages, temperature)
        if content is not None:
            return content
        last_error = error

        is_daily_quota_exhausted = error and (
            "free-models-per-day" in error or "rate_limit_exceeded" in error
        )
        is_transient_rate_limit = (
            not is_daily_quota_exhausted
            and error
            and any(code in error for code in ("429", "502", "ResourceExhausted"))
        )

        if is_transient_rate_limit:
            print(f"[OBSERVE] [{provider}] {model} rate-limited — retrying once after "
                  f"{_RETRY_WAIT_SECONDS}s before falling back ...")
            time.sleep(_RETRY_WAIT_SECONDS)
            content, error = _try_model(url, api_key, model, messages, temperature)
            if content is not None:
                return content
            last_error = error
        elif is_daily_quota_exhausted:
            print(f"[OBSERVE] [{provider}] {model} hit its daily free-tier quota — "
                  f"skipping retry, falling back to next model ...")

        print(f"[OBSERVE] [{provider}] {model} failed, falling back to next model ...")

    raise RuntimeError(f"All free models across all providers failed. Last error: {last_error}")
