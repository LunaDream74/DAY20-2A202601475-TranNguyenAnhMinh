from pathlib import Path

from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.storage import LocalArtifactStore
from multi_agent_research_lab.utils.timer import elapsed_timer


def test_artifact_store_creates_nested_text_file() -> None:
    root = Path("reports/gold/annotations/test_store")
    path = LocalArtifactStore(root).write_text("nested/report.md", "report")

    try:
        assert path.read_text(encoding="utf-8") == "report"
    finally:
        path.unlink(missing_ok=True)


def test_timer_and_logging_helpers() -> None:
    configure_logging("INFO")
    with elapsed_timer() as elapsed:
        sum(range(10))

    assert elapsed() >= 0
