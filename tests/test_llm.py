"""
Unit tests for tools/llm.py — the OpenRouter client with free-model
fallback and retry. All HTTP calls are mocked here; the real network
smoke test lives in tests/test_smoke_openrouter.py.
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
    monkeypatch.setattr(llm, "FREE_MODELS", ["model-a:free", "model-b:free"])

    fail = _fake_response(429, {"error": {"message": "rate limited"}})
    with patch("tools.llm.requests.post", return_value=fail):
        try:
            llm.call_llm([{"role": "user", "content": "hi"}])
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "All free models failed" in str(exc)


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
    test_handles_200_with_empty_choices()
    test_handles_network_exception()
    print("All llm tests passed.")
