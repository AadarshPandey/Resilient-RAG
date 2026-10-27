"""Application configuration and environment settings."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Resilient RAG loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        # Mirror Gemini key to GOOGLE_API_KEY if needed by langchain-google-genai / google client
        if self.gemini_api_key:
            if not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
            if not os.environ.get("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        # Mirror LangSmith tracing variables into os.environ for client tracing
        if self.langsmith_tracing:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if self.langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
                os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            if self.langsmith_api_key:
                if not os.environ.get("LANGSMITH_API_KEY"):
                    os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
                if not os.environ.get("LANGCHAIN_API_KEY"):
                    os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            if not os.environ.get("LANGSMITH_ENDPOINT"):
                os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
            if not os.environ.get("LANGCHAIN_ENDPOINT"):
                os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint

    # Core healing & recovery
    healing_enabled: bool = True
    max_heal_attempts: int = 4

    # LLM chain configuration (tier-0, tier-1)
    # Defaulting to Gemini models as requested by the user
    llm_chain: str = "google:gemini-2.5-flash,google:gemini-2.5-pro"

    # API Keys (accepts standard provider names as well as RAG_ prefixed)
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "RAG_GEMINI_API_KEY"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY"),
    )

    # Device & deep learning models
    device: str = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Storage paths
    qdrant_url: str | None = None
    qdrant_path: str = "data/qdrant"
    checkpoint_dsn: str = "sqlite:///data/checkpoints.db"
    audit_log_path: str = "data/audit.jsonl"
    corpus_dir: str = "data/raw"
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "RAG_REDIS_URL"),
    )

    # API & concurrency
    api_concurrency_limit: int = 10
    api_timeout_seconds: float = 30.0

    # Observability & tracing
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGSMITH_TRACING", "RAG_LANGSMITH_TRACING"),
    )
    langsmith_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "RAG_LANGSMITH_API_KEY"),
    )
    langsmith_project: str = Field(
        default="resilient-rag",
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "RAG_LANGSMITH_PROJECT"),
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices(
            "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT", "RAG_LANGSMITH_ENDPOINT"
        ),
    )

    def get_llm_tiers(self) -> list[tuple[str, str]]:
        """Parse RAG_LLM_CHAIN into a list of (provider, model_name) tuples."""
        tiers: list[tuple[str, str]] = []
        raw_entries = [entry.strip() for entry in self.llm_chain.split(",") if entry.strip()]
        for entry in raw_entries:
            if ":" in entry:
                provider, model = entry.split(":", 1)
                tiers.append((provider.strip().lower(), model.strip()))
            else:
                # Default provider is google
                tiers.append(("google", entry.strip()))
        return tiers

    def get_effective_gemini_api_key(self) -> str | None:
        """Return the effective Gemini / Google API key."""
        return self.gemini_api_key

    def ensure_directories(self) -> None:
        """Create necessary runtime directories."""
        Path(self.qdrant_path).mkdir(parents=True, exist_ok=True)
        Path(self.audit_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.corpus_dir).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Return an application Settings instance."""
    load_dotenv()
    return Settings()
