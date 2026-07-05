"""LLM access via LiteLLM/OpenRouter, with retries, timing, and typed errors.

Failures raise :class:`LLMError` instead of returning error text, so callers
can never mistake an error message for model output (e.g. store it as a
long-term memory). Every call is logged with its purpose and latency, and
recorded in the session trace when a tracer is attached.
"""

from __future__ import annotations

import logging
import time

import litellm

from personica.config import Settings
from personica.tracing import Tracer

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable response."""


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.tracer = tracer
        # Session-cumulative usage, surfaced at session end and in traces.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.chat_model)

    def _trace(self, event: str, **fields) -> None:
        if self.tracer is not None:
            self.tracer.event(event, **fields)

    def _record_usage(self, response) -> tuple[int, int, float]:
        """Accumulate token counts and cost from a litellm response."""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            cost = float(
                litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # unknown model in litellm's cost map, etc.
            cost = 0.0
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost
        return prompt_tokens, completion_tokens, cost

    def _complete(
        self,
        model: str,
        messages: list[Message],
        temperature: float,
        purpose: str,
    ) -> str:
        if not self.is_configured:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        attempts = 1 + self.max_retries
        last_error: Exception | None = None
        for attempt in range(attempts):
            start = time.monotonic()
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    api_key=self.settings.api_key,
                    api_base=self.settings.base_url,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    latency_ms = int((time.monotonic() - start) * 1000)
                    prompt_tokens, completion_tokens, cost = (
                        self._record_usage(response))
                    logger.debug(
                        "LLM %s call to %s succeeded in %dms "
                        "(attempt %d/%d, %d+%d tokens, ~$%.6f)",
                        purpose, model, latency_ms, attempt + 1, attempts,
                        prompt_tokens, completion_tokens, cost,
                    )
                    self._trace(
                        "llm_call",
                        purpose=purpose, model=model, ok=True,
                        latency_ms=latency_ms, attempts=attempt + 1,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=round(cost, 6),
                    )
                    return content.strip()
                last_error = LLMError(f"{model} returned an empty response")
            except Exception as exc:  # litellm raises provider-specific errors
                last_error = exc
            logger.warning(
                "LLM %s call to %s failed (attempt %d/%d): %s",
                purpose, model, attempt + 1, attempts, last_error,
            )
            if attempt < attempts - 1:
                time.sleep(self.retry_backoff_seconds * (attempt + 1))

        self._trace(
            "llm_call",
            purpose=purpose, model=model, ok=False,
            attempts=attempts, error=str(last_error),
        )
        raise LLMError(
            f"LLM {purpose} call to {model} failed after {attempts} attempts"
        ) from last_error

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.2,
        purpose: str = "chat",
    ) -> str:
        """Full-quality chat completion on the main model."""
        return self._complete(self.settings.chat_model, messages, temperature, purpose)

    def complete(
        self,
        prompt: str,
        *,
        use_utility_model: bool = True,
        temperature: float = 0.0,
        purpose: str = "utility",
    ) -> str:
        """Single-prompt completion; defaults to the cheaper utility model."""
        model = (
            self.settings.utility_model
            if use_utility_model
            else self.settings.chat_model
        )
        return self._complete(
            model, [{"role": "user", "content": prompt}], temperature, purpose)
