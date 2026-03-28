from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings, missing_secret_message


class TestSettings:
    def test_missing_secret_message_mentions_streamlit_cloud(self):
        message = missing_secret_message("OPENAI_API_KEY")
        assert "OPENAI_API_KEY" in message
        assert "Streamlit Community Cloud secrets" in message

    def test_streamlit_cloud_uses_tmp_chroma_dir(self, monkeypatch):
        monkeypatch.setenv("STREAMLIT_CLOUD", "1")
        settings = Settings()

        resolved = settings.resolved_chroma_dir()

        assert "/tmp" in str(resolved)
        assert resolved.name == "chroma_db"

    def test_local_default_chroma_dir_stays_in_repo(self, monkeypatch):
        monkeypatch.delenv("STREAMLIT_CLOUD", raising=False)
        monkeypatch.delenv("STREAMLIT_RUNTIME", raising=False)
        monkeypatch.delenv("STREAMLIT_SHARING_MODE", raising=False)
        settings = Settings()

        resolved = settings.resolved_chroma_dir()

        assert resolved == Path(__file__).resolve().parents[1] / "data" / "chroma_db"

    def test_is_ready_for_rag_requires_active_provider_key(self):
        settings = Settings(llm_provider="openai", openai_api_key="test-key")
        assert settings.is_ready_for_rag() is True

        missing = Settings(llm_provider="openai", openai_api_key="")
        assert missing.is_ready_for_rag() is False
