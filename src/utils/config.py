"""Configuration management for the shipping price search tool."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Configuration follows the precedence: environment variables -> .env file -> defaults.
    All sensitive values (API keys) must be provided via environment or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for embeddings and LLM operations",
    )

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use",
    )
    llm_model: str = Field(
        default="gpt-4o-2024-11-20",
        description="OpenAI LLM model for query processing",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (0.0 = deterministic)",
    )

    # Storage Paths
    pdf_storage_path: Path = Field(
        default=Path("data/pdfs"),
        description="Directory for uploaded PDF files",
    )
    vector_store_path: Path = Field(
        default=Path("data/storage"),
        description="Directory for FAISS vector store persistence",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host binding",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="FastAPI port",
    )

    # Query Configuration
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to retrieve per query",
    )
    query_parser_cache_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum number of parsed queries to cache (LRU cache for LLM query parser)",
    )
    table_parser_cache_size: int = Field(
        default=10000,
        ge=100,
        le=50000,
        description="Maximum number of table price extractions to cache (LRU cache for LLM table parser)",
    )
    chunk_size: int = Field(
        default=2048,
        ge=128,
        le=4096,
        description="Text chunk size for document indexing",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=512,
        description="Overlap between text chunks",
    )

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize settings and ensure required directories exist."""
        super().__init__(**kwargs)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.pdf_storage_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def load_settings(env_file: str | Path | None = None, reload: bool = False) -> Settings:
    """Load application settings from environment.

    Args:
        env_file: Optional path to .env file. If None, searches for .env in current directory.
        reload: If True, forces reload of settings even if already loaded.

    Returns:
        Settings instance with loaded configuration.

    Raises:
        ConfigError: If required configuration values are missing or invalid.
    """
    global _settings

    if _settings is not None and not reload:
        return _settings

    # Load .env file if it exists
    if env_file:
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv(override=False)

    try:
        _settings = Settings()
        return _settings
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}") from e


def get_settings() -> Settings:
    """Get the current settings instance.

    Returns:
        Current Settings instance.

    Raises:
        ConfigError: If settings have not been loaded yet.
    """
    if _settings is None:
        raise ConfigError(
            "Settings not loaded. Call load_settings() first or ensure .env file exists."
        )
    return _settings
