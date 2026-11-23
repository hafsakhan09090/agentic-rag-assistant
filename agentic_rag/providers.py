"""LLM providers (standard library only) and JSON extraction."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional


class ProviderError(Exception):
    pass


class Provider:
    name = "base"
    model = ""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


def extract_json(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ProviderError("no JSON object in model output")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProviderError("model output is not a JSON object")
    return obj


def _post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ProviderError(f"HTTP {exc.code} from model API: {exc.read().decode('utf-8', 'replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"could not reach model API: {exc.reason}") from exc


class OpenAICompatibleProvider(Provider):
    name = "openai"

    def __init__(self, model: Optional[str] = None, timeout: int = 60):
        self.api_key = os.environ.get("AGENT_API_KEY") or os.environ.get("LOGLENS_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (os.environ.get("AGENT_BASE_URL") or os.environ.get("LOGLENS_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("AGENT_MODEL") or os.environ.get("LOGLENS_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError("set AGENT_API_KEY (or LOGLENS_API_KEY) to use the 'openai' provider")

    def complete(self, system: str, user: str) -> str:
        data = _post(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {"model": self.model, "temperature": 0.1,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            self.timeout,
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("unexpected response shape from model API") from exc


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, timeout: int = 60):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("AGENT_MODEL") or "claude-sonnet-5-5"
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError("set ANTHROPIC_API_KEY to use the 'anthropic' provider")

    def complete(self, system: str, user: str) -> str:
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": self.model, "max_tokens": 1000, "temperature": 0.1, "system": system,
             "messages": [{"role": "user", "content": user}]},
            self.timeout,
        )
        try:
            return "".join(block.get("text", "") for block in data["content"])
        except (KeyError, TypeError) as exc:
            raise ProviderError("unexpected response shape from model API") from exc


def get_provider(name: str, model: Optional[str] = None) -> Optional[Provider]:
    """'scripted' returns None: the offline planner is used instead of a model."""
    if name == "scripted":
        return None
    if name == "openai":
        return OpenAICompatibleProvider(model)
    if name == "anthropic":
        return AnthropicProvider(model)
    raise ProviderError(f"unknown provider '{name}' (use scripted, openai or anthropic)")
