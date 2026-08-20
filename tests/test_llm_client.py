from typing import Any

import pytest
from pydantic import BaseModel

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.llm_client import LLMClient


class StructuredAnswer(BaseModel):
    value: str


class FakeUsage:
    input_tokens = 10
    output_tokens = 5


class FakeResponse:
    output_text = '{"value":"ok"}'
    usage = FakeUsage()

    def model_dump(self) -> dict[str, Any]:
        return {
            "output": [
                {
                    "action": {
                        "sources": [{"url": "https://a.example", "title": "A", "snippet": "A fact"}]
                    },
                    "content": [
                        {
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://b.example",
                                    "title": "B",
                                }
                            ]
                        }
                    ],
                }
            ]
        }


class FakeResponses:
    def __init__(self, *, fails: bool = False) -> None:
        self.request: dict[str, Any] = {}
        self.fails = fails

    def create(self, **request: Any) -> FakeResponse:
        if self.fails:
            raise RuntimeError("provider down")
        self.request = request
        return FakeResponse()


class FakeSDK:
    def __init__(self, *, fails: bool = False) -> None:
        self.responses = FakeResponses(fails=fails)


class FakeChatUsage:
    prompt_tokens = 7
    completion_tokens = 3


class FakeMessage:
    content = "Mistral candidate answer"


class FakeChoice:
    message = FakeMessage()


class FakeChatResponse:
    choices = [FakeChoice()]
    usage = FakeChatUsage()


class FakeCompletions:
    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def create(self, **request: Any) -> FakeChatResponse:
        self.request = request
        return FakeChatResponse()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeOpenRouterSDK:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_complete_tracks_usage_cost_and_provider_sources() -> None:
    sdk = FakeSDK()
    settings = Settings(_env_file=None, OPENAI_MODEL="gpt-4o-mini")
    client = LLMClient(settings, sdk)

    response = client.complete("system", "question", web_search=True)

    assert response.input_tokens == 10
    assert response.cost_usd == pytest.approx(0.0000045)
    assert [source.url for source in response.sources] == [
        "https://a.example",
        "https://b.example",
    ]
    assert sdk.responses.request["tool_choice"] == "required"


def test_complete_structured_sends_strict_json_schema() -> None:
    sdk = FakeSDK()
    client = LLMClient(Settings(_env_file=None), sdk)

    parsed, usage = client.complete_structured("judge", "input", StructuredAnswer)

    assert parsed.value == "ok"
    assert usage.output_tokens == 5
    assert sdk.responses.request["text"]["format"]["type"] == "json_schema"
    assert sdk.responses.request["text"]["format"]["strict"] is True


def test_provider_errors_are_wrapped() -> None:
    client = LLMClient(Settings(_env_file=None), FakeSDK(fails=True))

    with pytest.raises(AgentExecutionError, match="OpenAI request failed"):
        client.complete("system", "input")
    with pytest.raises(AgentExecutionError, match="Structured OpenAI request failed"):
        client.complete_structured("system", "input", StructuredAnswer)


def test_openrouter_uses_chat_completions_for_candidate_generation() -> None:
    sdk = FakeOpenRouterSDK()
    client = LLMClient(
        Settings(_env_file=None),
        sdk,
        provider="openrouter",
        model="mistralai/mistral-7b-instruct:free",
    )

    response = client.complete("system role", "frozen evidence")

    assert response.content == "Mistral candidate answer"
    assert response.input_tokens == 7
    assert response.output_tokens == 3
    assert response.cost_usd is None
    assert sdk.chat.completions.request["model"] == "mistralai/mistral-7b-instruct:free"
    assert sdk.chat.completions.request["messages"][0]["role"] == "system"


def test_openrouter_rejects_openai_specific_features() -> None:
    client = LLMClient(Settings(_env_file=None), FakeOpenRouterSDK(), provider="openrouter")

    with pytest.raises(AgentExecutionError, match="cannot use OpenAI built-in web search"):
        client.complete("system", "input", web_search=True)
    with pytest.raises(AgentExecutionError, match="require OpenAI"):
        client.complete_structured("system", "input", StructuredAnswer)
