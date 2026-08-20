from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openai_model
    assert settings.system_provider in {"openai", "openrouter"}
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.eval_judge_provider == "openai"
    assert settings.max_iterations >= 1
