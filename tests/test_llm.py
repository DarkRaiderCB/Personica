"""LLMClient tests — litellm.completion is monkeypatched, no network."""

from types import SimpleNamespace

import litellm
import pytest

from conftest import make_settings
from personica.llm import LLMClient, LLMError


def make_response(content, prompt_tokens=0, completion_tokens=0):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def make_client(**kwargs) -> LLMClient:
    return LLMClient(make_settings(**kwargs), retry_backoff_seconds=0)


def test_chat_success(monkeypatch):
    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: make_response("  hello  "))
    client = make_client()
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"


def test_not_configured_raises_without_calling_api(monkeypatch):
    def explode(**kw):
        raise AssertionError("API must not be called")

    monkeypatch.setattr(litellm, "completion", explode)
    client = make_client(api_key="")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return make_response("recovered")

    monkeypatch.setattr(litellm, "completion", flaky)
    client = make_client()
    assert client.complete("prompt") == "recovered"
    assert calls["n"] == 2


def test_empty_responses_retry_then_raise(monkeypatch):
    calls = {"n": 0}

    def empty(**kw):
        calls["n"] += 1
        return make_response("")

    monkeypatch.setattr(litellm, "completion", empty)
    client = make_client()
    with pytest.raises(LLMError):
        client.complete("prompt")
    assert calls["n"] == 3  # 1 attempt + 2 retries


def test_exhausted_retries_raise_llm_error(monkeypatch):
    def always_fail(**kw):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "completion", always_fail)
    client = make_client()
    with pytest.raises(LLMError, match="failed after 3 attempts"):
        client.chat([{"role": "user", "content": "hi"}])


def test_model_selection_and_credentials(monkeypatch):
    captured = []

    def capture(**kw):
        captured.append(kw)
        return make_response("ok")

    monkeypatch.setattr(litellm, "completion", capture)
    client = make_client()

    client.complete("p")  # utility model by default
    client.complete("p", use_utility_model=False)
    client.chat([{"role": "user", "content": "hi"}])

    assert captured[0]["model"] == "test/utility-model"
    assert captured[1]["model"] == "test/chat-model"
    assert captured[2]["model"] == "test/chat-model"
    for kw in captured:
        assert kw["api_key"] == "test-key"
        assert kw["api_base"] == "https://example.invalid/api/v1"


def test_usage_and_cost_accumulate_across_calls(monkeypatch):
    monkeypatch.setattr(
        litellm, "completion",
        lambda **kw: make_response("ok", prompt_tokens=100, completion_tokens=20))
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0015)
    client = make_client()

    client.complete("p1")
    client.complete("p2")

    assert client.total_prompt_tokens == 200
    assert client.total_completion_tokens == 40
    assert client.total_cost_usd == pytest.approx(0.003)


def test_unknown_cost_model_does_not_break_calls(monkeypatch):
    def cost_explodes(**kw):
        raise ValueError("model not in cost map")

    monkeypatch.setattr(
        litellm, "completion",
        lambda **kw: make_response("ok", prompt_tokens=10, completion_tokens=5))
    monkeypatch.setattr(litellm, "completion_cost", cost_explodes)
    client = make_client()

    assert client.complete("p") == "ok"
    assert client.total_prompt_tokens == 10
    assert client.total_cost_usd == 0.0


def test_tracer_records_success_and_failure(monkeypatch):
    events = []

    class SpyTracer:
        def event(self, event, **fields):
            events.append({"event": event, **fields})

    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: make_response("ok"))
    client = LLMClient(make_settings(), retry_backoff_seconds=0,
                       tracer=SpyTracer())
    client.complete("p", purpose="query_rewrite")
    assert events[-1]["event"] == "llm_call"
    assert events[-1]["purpose"] == "query_rewrite"
    assert events[-1]["ok"] is True
    assert "latency_ms" in events[-1]

    def always_fail(**kw):
        raise RuntimeError("down")

    monkeypatch.setattr(litellm, "completion", always_fail)
    with pytest.raises(LLMError):
        client.complete("p", purpose="relevance_check")
    assert events[-1]["ok"] is False
    assert events[-1]["purpose"] == "relevance_check"
