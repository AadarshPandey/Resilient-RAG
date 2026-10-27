"""Application configuration and environment settings."""

from pathlib import Path

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

    # Core healing & recovery
    healing_enabled: bool = True
    max_heal_attempts: int = 4

    # LLM chain configuration (tier-0, tier-1)
    # Defaulting to Gemini models as requested by the user
    llm_chain: str = "google:gemini-1.5-flash,google:gemini-1.5-pro"

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
