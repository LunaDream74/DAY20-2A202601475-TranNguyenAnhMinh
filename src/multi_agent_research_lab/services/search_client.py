"""Search client abstraction for ResearcherAgent."""

import re
from dataclasses import dataclass

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClient


@dataclass(frozen=True)
class SearchResponse:
    """Grounded search synthesis plus its provider metadata."""

    synthesis: str
    sources: list[SourceDocument]
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class SearchClient:
    """Web search implemented with OpenAI's built-in search tool.

    This path needs no Tavily key. Replacing this class is enough to switch providers.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        """Search and return a cited synthesis with up to ``max_results`` URLs."""

        response = self.llm_client.complete(
            "You are a careful web researcher. Prefer primary and recent sources. "
            "Separate sourced facts from uncertainty and never invent a URL.",
            f"Research this question: {query}\n"
            f"Use at most {max_results} strong sources. Return concise factual notes with "
            "citations.",
            web_search=True,
        )
        sources = response.sources[:max_results]
        return SearchResponse(
            synthesis=self._keep_grounded_blocks(response.content, sources),
            sources=sources,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

    @staticmethod
    def _keep_grounded_blocks(content: str, sources: list[SourceDocument]) -> str:
        """Drop paragraphs that cite only URLs excluded by the source limit."""

        allowed = {source.url for source in sources if source.url}
        blocks: list[str] = []
        for block in content.split("\n\n"):
            urls = set(re.findall(r"https?://[^\s)]+", block))
            if not urls or urls <= allowed:
                blocks.append(block)
        return "\n\n".join(blocks)
