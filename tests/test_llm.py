"""
Unit tests for tools/llm.py — the two-tier Groq-then-OpenRouter free
client with model fallback and retry. All HTTP calls are mocked here;
the real network smoke test lives in tests/test_smoke_openrouter.py.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.llm as llm


def _fake_response(status_code, json_body):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = str(json_body)
    return resp


def test_first_model_succeeds_immediately():
    ok_response = _fake_response(200, {"choices": [{"message": {"content": "hello"}}]})
    with patch("tools.llm.requests.post", return_value=ok_response) as mock_post:
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "hello"
        assert mock_post.call_count == 1


def test_falls_back_to_second_model_on_404():
    responses = [
        _fake_response(404, {"error": {"message": "not found"}}),
        _fake_response(200, {"choices": [{"message": {"content": "second model answer"}}]}),
    ]
    with patch("tools.llm.requests.post", side_effect=responses):
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "second model answer"


def test_retries_once_on_429_before_falling_back(monkeypatch):
    # Speed up the test — don't actually sleep 8 seconds.
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    responses = [
        _fake_response(429, {"error": {"message": "rate limited"}}),  # first try
        _fake_response(429, {"error": {"message": "rate limited"}}),  # retry, still limited
        _fake_response(200, {"choices": [{"message": {"content": "third call answer"}}]}),
    ]
    with patch("tools.llm.requests.post", side_effect=responses) as mock_post:
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "third call answer"
        assert mock_post.call_count == 3


def test_recovers_immediately_if_retry_succeeds(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)

    responses = [
        _fake_response(429, {"error": {"message": "rate limited"}}),  # first try fails
        _fake_response(200, {"choices": [{"message": {"content": "retry succeeded"}}]}),  # retry succeeds
    ]
    with patch("tools.llm.requests.post", side_effect=responses) as mock_post:
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "retry succeeded"
        assert mock_post.call_count == 2


def test_raises_when_all_models_exhausted(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [
        ("groq", llm.GROQ_URL, "fake-groq-key", "model-a"),
        ("openrouter", llm.OPENROUTER_URL, "fake-or-key", "model-b:free"),
    ])

    fail = _fake_response(429, {"error": {"message": "rate limited"}})
    with patch("tools.llm.requests.post", return_value=fail):
        try:
            llm.call_llm([{"role": "user", "content": "hi"}])
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "All free models across all providers failed" in str(exc)


def test_falls_back_from_groq_to_openrouter_across_providers(monkeypatch):
    """
    The core new behavior: when every Groq model fails, call_llm must
    continue into the OpenRouter models rather than stopping at the
    end of the Groq tier.
    """
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [
        ("groq", llm.GROQ_URL, "fake-groq-key", "groq-model-a"),
        ("groq", llm.GROQ_URL, "fake-groq-key", "groq-model-b"),
        ("openrouter", llm.OPENROUTER_URL, "fake-or-key", "or-model:free"),
    ])

    groq_fail = _fake_response(404, {"error": {"message": "model not found"}})
    or_ok = _fake_response(200, {"choices": [{"message": {"content": "openrouter saved it"}}]})

    with patch("tools.llm.requests.post", side_effect=[groq_fail, groq_fail, or_ok]):
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "openrouter saved it"


def test_skips_provider_with_no_api_key_configured(monkeypatch):
    """If GROQ_API_KEY is unset (falsy), Groq entries in the chain must
    be skipped without making a request, falling straight to OpenRouter."""
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [
        ("groq", llm.GROQ_URL, None, "groq-model-a"),
        ("openrouter", llm.OPENROUTER_URL, "fake-or-key", "or-model:free"),
    ])

    or_ok = _fake_response(200, {"choices": [{"message": {"content": "openrouter only"}}]})
    with patch("tools.llm.requests.post", return_value=or_ok) as mock_post:
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "openrouter only"
        assert mock_post.call_count == 1  # groq entry skipped entirely, no HTTP call made


def test_groq_rate_limit_exceeded_treated_as_daily_quota_not_transient(monkeypatch):
    """Groq's daily-quota error message differs from OpenRouter's
    ("rate_limit_exceeded" vs "free-models-per-day") — both must skip
    the retry-and-wait path."""
    sleep_calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(llm, "_PROVIDER_CHAIN", [
        ("groq", llm.GROQ_URL, "fake-groq-key", "groq-model-a"),
        ("openrouter", llm.OPENROUTER_URL, "fake-or-key", "or-model:free"),
    ])

    groq_quota_exhausted = _fake_response(429, {
        "error": {"message": "rate_limit_exceeded: daily token limit reached", "type": "rate_limit_exceeded"}
    })
    or_ok = _fake_response(200, {"choices": [{"message": {"content": "fell back cleanly"}}]})

    with patch("tools.llm.requests.post", side_effect=[groq_quota_exhausted, or_ok]) as mock_post:
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "fell back cleanly"
        assert sleep_calls == []  # no wasted retry wait
        assert mock_post.call_count == 2  # one groq attempt, no retry, one openrouter attempt


def test_daily_quota_exhaustion_skips_retry_and_falls_back_immediately(monkeypatch):
    """
    A daily-quota 429 ("free-models-per-day") is not transient within
    this run — retrying it wastes _RETRY_WAIT_SECONDS for nothing, so
    call_llm should skip straight to the next model without sleeping.
    """
    sleep_calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    quota_exhausted = _fake_response(429, {
        "error": {
            "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day",
            "code": 429,
        }
    })
    ok_response = _fake_response(200, {"choices": [{"message": {"content": "next model answer"}}]})

    with patch("tools.llm.requests.post", side_effect=[quota_exhausted, ok_response]) as mock_post:
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "next model answer"
        assert mock_post.call_count == 2  # first model tried once, no retry, then fallback
        assert sleep_calls == []  # never slept — daily quota isn't worth retrying


def test_embedded_error_in_200_response_is_treated_as_failure():
    """
    Some providers (observed with an Nvidia-backed free model) return
    HTTP 200 with an error embedded in the JSON body instead of a real
    error status code. call_llm must detect this and fall back.
    """
    embedded_error = _fake_response(200, {
        "id": "gen-123",
        "error": {"message": "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)", "code": 502},
    })
    ok_response = _fake_response(200, {"choices": [{"message": {"content": "fallback worked"}}]})

    with patch("tools.llm.requests.post", side_effect=[embedded_error, ok_response]):
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "fallback worked"


def test_handles_200_with_empty_choices():
    empty_choices = _fake_response(200, {"choices": []})
    ok_response = _fake_response(200, {"choices": [{"message": {"content": "fallback ok"}}]})
    with patch("tools.llm.requests.post", side_effect=[empty_choices, ok_response]):
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "fallback ok"


def test_handles_network_exception():
    import requests as real_requests

    ok_response = _fake_response(200, {"choices": [{"message": {"content": "recovered"}}]})
    with patch(
        "tools.llm.requests.post",
        side_effect=[real_requests.exceptions.ConnectionError("boom"), ok_response],
    ):
        result = llm.call_llm([{"role": "user", "content": "hi"}])
        assert result == "recovered"


if __name__ == "__main__":
    test_first_model_succeeds_immediately()
    test_falls_back_to_second_model_on_404()

    class _Ctx:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    ctx = _Ctx()
    test_retries_once_on_429_before_falling_back(ctx)
    test_recovers_immediately_if_retry_succeeds(ctx)
    test_raises_when_all_models_exhausted(ctx)
    test_falls_back_from_groq_to_openrouter_across_providers(ctx)
    test_skips_provider_with_no_api_key_configured(ctx)
    test_groq_rate_limit_exceeded_treated_as_daily_quota_not_transient(ctx)
    test_daily_quota_exhaustion_skips_retry_and_falls_back_immediately(ctx)
    test_embedded_error_in_200_response_is_treated_as_failure()
    test_handles_200_with_empty_choices()
    test_handles_network_exception()
    print("All llm tests passed.")
