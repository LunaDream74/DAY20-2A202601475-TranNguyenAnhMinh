"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.observability.tracing import configure_provider_tracing

_DEFAULT_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    sources: list[SourceDocument] = field(default_factory=list)


class LLMClient:
    """Provider adapter for OpenAI Responses and OpenRouter Chat Completions."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        *,
        provider: Literal["openai", "openrouter"] = "openai",
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider
        self.model = model or self.settings.openai_model
        if client is not None:
            self._client = client
            return
        api_key = (
            self.settings.openrouter_api_key
            if provider == "openrouter"
            else self.settings.openai_api_key
        )
        key_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
        if not api_key:
            raise AgentExecutionError(f"{key_name} is required for {provider} LLM calls")
        configure_provider_tracing(self.settings)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentExecutionError(
                'OpenAI SDK is missing; run: pip install -e ".[dev,llm]"'
            ) from exc
        client_options: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": self.settings.openai_max_retries,
            "timeout": self.settings.openai_request_timeout_seconds,
        }
        if provider == "openrouter":
            client_options["base_url"] = self.settings.openrouter_base_url
        openai_client = OpenAI(
            **client_options,
        )
        if self.settings.langsmith_api_key:
            try:
                from langsmith.wrappers import wrap_openai

                openai_client = wrap_openai(openai_client)
            except ImportError:
                pass
        self._client = openai_client

    def complete(
        self, system_prompt: str, user_prompt: str, *, web_search: bool = False
    ) -> LLMResponse:
        """Return one completion, optionally allowing OpenAI web search.

        Retry, timeout, usage extraction, citation extraction, and cost estimation live here
        so agent code remains provider-independent.
        """

        if self.provider == "openrouter":
            return self._complete_chat(system_prompt, user_prompt, web_search=web_search)

        request: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
        }
        if web_search:
            request["tools"] = [{"type": "web_search"}]
            request["tool_choice"] = "required"
            request["include"] = ["web_search_call.action.sources"]
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            raise AgentExecutionError(f"OpenAI request failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        return LLMResponse(
            content=response.output_text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
            sources=self._extract_sources(response),
        )

    def _complete_chat(
        self, system_prompt: str, user_prompt: str, *, web_search: bool
    ) -> LLMResponse:
        """Call OpenRouter's OpenAI-compatible Chat Completions endpoint."""

        if web_search:
            raise AgentExecutionError(
                "OpenRouter system models cannot use OpenAI built-in web search; "
                "use frozen evidence or a separate search provider"
            )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        except Exception as exc:
            raise AgentExecutionError(f"OpenRouter request failed: {exc}") from exc
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return LLMResponse(
            content=(content or "").strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> tuple[StructuredModel, LLMResponse]:
        """Return a schema-validated response for bounded evaluation judgments."""

        if self.provider != "openai":
            raise AgentExecutionError("Structured gold judgments currently require OpenAI")

        request: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__.lower(),
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                }
            },
        }
        try:
            response = self._client.responses.create(**request)
            parsed = response_model.model_validate_json(response.output_text)
        except Exception as exc:
            raise AgentExecutionError(f"Structured OpenAI request failed: {exc}") from exc
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        metadata = LLMResponse(
            content=response.output_text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens, output_tokens),
        )
        return parsed, metadata

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if self.provider != "openai":
            return None
        rates = _DEFAULT_PRICING_PER_MILLION.get(self.model)
        input_rate = self.settings.openai_input_cost_per_million
        output_rate = self.settings.openai_output_cost_per_million
        if input_rate is None and rates:
            input_rate = rates[0]
        if output_rate is None and rates:
            output_rate = rates[1]
        if input_rate is None or output_rate is None:
            return None
        return ((input_tokens or 0) * input_rate + (output_tokens or 0) * output_rate) / 1_000_000

    @staticmethod
    def _extract_sources(response: Any) -> list[SourceDocument]:
        """Extract URL citation annotations without depending on SDK model classes."""

        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        found: dict[str, SourceDocument] = {}
        for item in raw.get("output", []):
            for source in item.get("action", {}).get("sources", []):
                url = source.get("url")
                if not url or url in found:
                    continue
                found[url] = SourceDocument(
                    title=source.get("title") or url,
                    url=url,
                    snippet=source.get("snippet") or "Returned by OpenAI web search.",
                    metadata={"provider": "openai_web_search"},
                )
            for content in item.get("content", []):
                for annotation in content.get("annotations", []):
                    if annotation.get("type") != "url_citation":
                        continue
                    url = annotation.get("url")
                    if not url or url in found:
                        continue
                    found[url] = SourceDocument(
                        title=annotation.get("title") or url,
                        url=url,
                        snippet="Referenced by the OpenAI web-search response.",
                        metadata={"provider": "openai_web_search"},
                    )
        return list(found.values())
