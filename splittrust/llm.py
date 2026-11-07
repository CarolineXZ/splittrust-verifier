"""Chat backends behind one tiny interface: `complete(messages, tools=None, temperature=0) -> ChatResponse`.

- `OpenAICompatible` talks to any /v1/chat/completions endpoint: OpenAI (GPT-4o), Ollama (localhost:11434/v1),
  vLLM (localhost:8000/v1), and most hosted open-model APIs. The actor uses its tool-calling.
- `AnthropicMessages` is an optional backend for the verifier role (no tool calling needed there).
- `ScriptedBackend` returns canned responses for tests.
- `AuditedBackend` wraps any backend and scans every outbound request for canary strings. Wrap the
  frontier-bound backends with it; the count of hits is the privacy metric.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .canary import CanaryRegistry
from .env import ToolCall


@dataclass
class ChatResponse:
    text: str = ""
    tool_calls: list[tuple[str, ToolCall]] = field(default_factory=list)  # (call_id, call)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    assistant_message: dict[str, Any] = field(default_factory=dict)  # provider-native, to append to history


class OpenAICompatible:
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "ollama")
        self.timeout = timeout

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0) -> ChatResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                       headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append((tc["id"], ToolCall(fn["name"], args)))
        usage = d.get("usage") or {}
        return ChatResponse(text=msg.get("content") or "", tool_calls=calls, prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0), latency_ms=(time.perf_counter() - t0) * 1000,
                            assistant_message=msg)

    @staticmethod
    def tool_result_message(call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "content": content}


class AnthropicMessages:
    """Verifier-only backend (no tool calling). Uses the Messages API directly."""

    def __init__(self, model: str, api_key: str | None = None, timeout: float = 60.0, max_tokens: int = 400):
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0) -> ChatResponse:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        t0 = time.perf_counter()
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=self.timeout,
                       headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                       json={"model": self.model, "max_tokens": self.max_tokens, "temperature": temperature, "system": system,
                             "messages": rest})
        r.raise_for_status()
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage") or {}
        return ChatResponse(text=text, prompt_tokens=u.get("input_tokens", 0), completion_tokens=u.get("output_tokens", 0),
                            latency_ms=(time.perf_counter() - t0) * 1000, assistant_message={"role": "assistant", "content": text})


class ScriptedBackend:
    """Returns whatever `fn(messages, tools)` says. For tests."""

    def __init__(self, fn: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], ChatResponse]):
        self.fn = fn
        self.requests: list[dict[str, Any]] = []

    def complete(self, messages, tools=None, temperature=0.0) -> ChatResponse:
        self.requests.append({"messages": messages, "tools": tools})
        return self.fn(messages, tools)

    @staticmethod
    def tool_result_message(call_id: str, content: str) -> dict[str, Any]:
        return OpenAICompatible.tool_result_message(call_id, content)


class AuditedBackend:
    """Wraps a backend that sits OUTSIDE the perimeter. Every request body is scanned for canaries."""

    def __init__(self, inner: Any, canaries: CanaryRegistry, channel: str, strict: bool = False):
        self.inner, self.canaries, self.channel, self.strict = inner, canaries, channel, strict
        self.hits: list[dict[str, Any]] = []
        self.requests = 0

    def complete(self, messages, tools=None, temperature=0.0) -> ChatResponse:
        self.requests += 1
        found = self.canaries.scan(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False))
        if found:
            self.hits.append({"channel": self.channel, "request": self.requests, "canaries": found})
            if self.strict:
                raise RuntimeError(f"canary {found} would leave the perimeter via {self.channel}")
        return self.inner.complete(messages, tools=tools, temperature=temperature)

    def tool_result_message(self, call_id: str, content: str) -> dict[str, Any]:
        return self.inner.tool_result_message(call_id, content)

    def __getattr__(self, item: str) -> Any:  # model name etc.
        return getattr(self.inner, item)


def backend_from_env(role: str) -> OpenAICompatible | AnthropicMessages:
    """Build a backend from environment variables. Roles: actor | frontier | local.

    ACTOR_MODEL / FRONTIER_MODEL / LOCAL_MODEL name the model; *_BASE_URL overrides the endpoint;
    LOCAL_BASE_URL defaults to Ollama. A model name starting with `claude` uses the Anthropic API.
    """
    defaults = {"actor": ("gpt-4o-mini", "https://api.openai.com/v1"), "frontier": ("gpt-4o", "https://api.openai.com/v1"),
                "local": ("llama3.1:8b", "http://localhost:11434/v1")}
    model = os.environ.get(f"{role.upper()}_MODEL", defaults[role][0])
    base = os.environ.get(f"{role.upper()}_BASE_URL", defaults[role][1])
    if model.startswith("claude"):
        return AnthropicMessages(model)
    key = os.environ.get(f"{role.upper()}_API_KEY") or os.environ.get("OPENAI_API_KEY") or "ollama"
    return OpenAICompatible(model, base_url=base, api_key=key)
