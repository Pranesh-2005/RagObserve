"""Pluggable LLM generation backends for the live "replay generation" feature.

RAGObserve is an observability tool, but it can also *re-run* a generation against
the exact context a trace captured — useful for debugging grounding, comparing
models, or filling in a generation that was never logged.

It is provider-agnostic by design (see idea.md). Rather than depend on a
different SDK per vendor, every backend goes through ``httpx`` (already a core
dependency): the broad OpenAI-compatible ``/chat/completions`` shape covers
OpenAI, Gemini, Groq, OpenRouter, Together, Mistral, DeepSeek, Fireworks,
Perplexity and any other compatible gateway, while Anthropic uses its native
``/v1/messages`` API and Ollama its local ``/api/chat``. Providers are detected
at runtime from environment variables so the tool stays zero-config and offline
by default.

Each backend returns a uniform dict::

    {"text": str, "input_tokens": int|None, "output_tokens": int|None,
     "model": str, "provider": str}

Cost is computed by the caller via ``pricing.estimate_cost``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

ANTHROPIC_VERSION = "2023-06-01"
_TIMEOUT = 120.0


class ProviderError(RuntimeError):
    """Raised when a provider is unavailable or a call fails."""


# Registry: id -> config. ``kind`` selects the wire protocol.
#   openai  -> POST {base}/chat/completions, Bearer auth (OpenAI-compatible)
#   anthropic -> POST {base}/messages, x-api-key auth
#   ollama  -> local server, no auth
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic (Claude)", "kind": "anthropic",
        "env": ["ANTHROPIC_API_KEY"], "base": "https://api.anthropic.com/v1",
        "default_model": "claude-opus-4-8",
    },
    "openai": {
        "label": "OpenAI", "kind": "openai", "env": ["OPENAI_API_KEY"],
        "base": "https://api.openai.com/v1", "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "label": "Google Gemini", "kind": "openai", "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
    },
    "groq": {
        "label": "Groq", "kind": "openai", "env": ["GROQ_API_KEY"],
        "base": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "label": "OpenRouter", "kind": "openai", "env": ["OPENROUTER_API_KEY"],
        "base": "https://openrouter.ai/api/v1", "default_model": "openai/gpt-4o-mini",
    },
    "together": {
        "label": "Together AI", "kind": "openai", "env": ["TOGETHER_API_KEY"],
        "base": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "mistral": {
        "label": "Mistral", "kind": "openai", "env": ["MISTRAL_API_KEY"],
        "base": "https://api.mistral.ai/v1", "default_model": "mistral-small-latest",
    },
    "deepseek": {
        "label": "DeepSeek", "kind": "openai", "env": ["DEEPSEEK_API_KEY"],
        "base": "https://api.deepseek.com/v1", "default_model": "deepseek-chat",
    },
    "fireworks": {
        "label": "Fireworks", "kind": "openai", "env": ["FIREWORKS_API_KEY"],
        "base": "https://api.fireworks.ai/inference/v1",
        "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "perplexity": {
        "label": "Perplexity", "kind": "openai", "env": ["PERPLEXITY_API_KEY"],
        "base": "https://api.perplexity.ai", "default_model": "sonar",
    },
    "ollama": {
        "label": "Ollama (local)", "kind": "ollama", "env": [],
        "base": None, "default_model": "llama3",
    },
}


def _api_key(cfg: Dict[str, Any]) -> Optional[str]:
    for name in cfg.get("env", []):
        val = os.getenv(name)
        if val:
            return val
    return None


# --------------------------------------------------------------------- backends

def _call_openai_compatible(cfg, model, system, prompt, max_tokens) -> Dict[str, Any]:
    key = _api_key(cfg)
    if not key:
        raise ProviderError(f"missing API key ({' or '.join(cfg['env'])})")
    base = os.getenv("RAGOBSERVE_OPENAI_BASE_URL", cfg["base"]).rstrip("/")
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ProviderError(f"{cfg['label']} returned {e.response.status_code}: "
                            f"{_err_body(e.response)}") from e
    except Exception as e:
        raise ProviderError(f"{cfg['label']} call failed: {e}") from e
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return {
        "text": (choice.get("message") or {}).get("content", "") or "",
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "model": data.get("model", model),
        "provider": cfg["id"],
    }


def _call_anthropic(cfg, model, system, prompt, max_tokens) -> Dict[str, Any]:
    key = _api_key(cfg)
    if not key:
        raise ProviderError("missing API key (ANTHROPIC_API_KEY)")
    base = cfg["base"].rstrip("/")
    body: Dict[str, Any] = {
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    try:
        r = httpx.post(
            f"{base}/messages",
            headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                     "Content-Type": "application/json"},
            json=body, timeout=_TIMEOUT,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ProviderError(f"Anthropic returned {e.response.status_code}: "
                            f"{_err_body(e.response)}") from e
    except Exception as e:
        raise ProviderError(f"Anthropic call failed: {e}") from e
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage") or {}
    return {
        "text": text,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "model": data.get("model", model),
        "provider": "anthropic",
    }


def _call_ollama(cfg, model, system, prompt, max_tokens) -> Dict[str, Any]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = httpx.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"num_predict": max_tokens}},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
    except Exception as e:
        raise ProviderError(f"Ollama call failed (is it running at {host}?): {e}") from e
    data = r.json()
    return {
        "text": (data.get("message") or {}).get("content", "") or "",
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "model": data.get("model", model),
        "provider": "ollama",
    }


def _err_body(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("error", resp.text))[:300]
    except Exception:
        return resp.text[:300]


_DISPATCH = {"openai": _call_openai_compatible, "anthropic": _call_anthropic, "ollama": _call_ollama}


# ------------------------------------------------------------------- public API

def available_providers() -> List[Dict[str, Any]]:
    """Report every provider plus whether it's usable right now (key present /
    local), so the UI can guide the user without leaking which keys are set."""
    out = []
    for pid, cfg in PROVIDERS.items():
        ready = cfg["kind"] == "ollama" or bool(_api_key(cfg))
        hint = (f"runs against {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}"
                if cfg["kind"] == "ollama" else f"set {' or '.join(cfg['env'])}")
        out.append({
            "id": pid, "label": cfg["label"], "ready": ready,
            "default_model": cfg["default_model"], "hint": hint,
        })
    return out


def generate(provider: str, model: str, prompt: str, system: Optional[str] = None,
             max_tokens: int = 1024) -> Dict[str, Any]:
    cfg = PROVIDERS.get((provider or "").lower())
    if cfg is None:
        raise ProviderError(f"unknown provider: {provider!r}")
    cfg = {**cfg, "id": provider.lower()}
    model = model or cfg["default_model"]
    return _DISPATCH[cfg["kind"]](cfg, model, system, prompt, max_tokens)
