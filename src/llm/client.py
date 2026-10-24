"""Clientes de LLM com uma interface única (`LLMClient.chat`).

- OllamaClient: modelo local (Llama 3.1, Qwen 2.5...) via API HTTP do Ollama — padrão
  do projeto: gratuito e os dados do paciente não saem da máquina.
- OpenAICompatibleClient: qualquer servidor no formato OpenAI (OpenAI, LM Studio,
  vLLM, Groq...). Permite trocar de provedor sem mudar o resto do código.
- FakeLLMClient: respostas determinísticas para testes automatizados.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    model: str

    def chat(self, messages: list[dict[str, str]], *, json_schema: dict | None = None,
             temperature: float = 0.2) -> LLMResponse: ...


class LLMError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    timeout: float = 180.0
    num_ctx: int = 8192

    def chat(self, messages, *, json_schema=None, temperature=0.2) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "seed": 42},
        }
        if json_schema is not None:
            # "Structured outputs" do Ollama: a geração é restrita ao JSON Schema.
            payload["format"] = json_schema
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Falha ao chamar Ollama em {self.base_url} (modelo {self.model}): {exc}") from exc
        data = resp.json()
        latency = time.perf_counter() - start
        result = LLMResponse(
            content=data["message"]["content"],
            model=self.model,
            latency_seconds=latency,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
        logger.info("llm_call", extra={"event": "llm_call", "provider": "ollama", "model": self.model,
                                       "latency_seconds": round(latency, 3), "prompt_tokens": result.prompt_tokens,
                                       "completion_tokens": result.completion_tokens})
        return result

    def is_available(self) -> bool:
        try:
            tags = requests.get(f"{self.base_url}/api/tags", timeout=3).json()
        except requests.RequestException:
            return False
        names = {m["name"] for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names


@dataclass
class OpenAICompatibleClient:
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    timeout: float = 120.0

    def chat(self, messages, *, json_schema=None, temperature=0.2) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "resposta", "schema": json_schema},
            }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Falha ao chamar {self.base_url} (modelo {self.model}): {exc}") from exc
        data = resp.json()
        latency = time.perf_counter() - start
        usage = data.get("usage", {})
        logger.info("llm_call", extra={"event": "llm_call", "provider": "openai_compatible", "model": self.model,
                                       "latency_seconds": round(latency, 3)})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=self.model,
            latency_seconds=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


@dataclass
class FakeLLMClient:
    """Cliente de teste: devolve `responder(messages)` ou uma resposta fixa."""
    response: str | dict = "{}"
    responder: Callable[[list[dict[str, str]]], str] | None = None
    model: str = "fake"
    calls: list = field(default_factory=list)

    def chat(self, messages, *, json_schema=None, temperature=0.2) -> LLMResponse:
        self.calls.append({"messages": messages, "json_schema": json_schema, "temperature": temperature})
        if self.responder is not None:
            content = self.responder(messages)
        elif isinstance(self.response, dict):
            content = json.dumps(self.response, ensure_ascii=False)
        else:
            content = self.response
        return LLMResponse(content=content, model=self.model, latency_seconds=0.0)


def build_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Fábrica configurada por variáveis de ambiente (LLM_PROVIDER=ollama|openai)."""
    provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        return OllamaClient(model=model) if model else OllamaClient()
    if provider == "openai":
        return OpenAICompatibleClient(model=model) if model else OpenAICompatibleClient()
    raise ValueError(f"Provedor de LLM desconhecido: {provider}")
