"""Application configuration for the RAG pipeline."""

from pydantic import computed_field
from pydantic.networks import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or a .env file.

    Attributes
    ----------
    postgres_user
        PostgreSQL username.
    postgres_password
        PostgreSQL password.
    postgres_host
        PostgreSQL host.
    postgres_port
        PostgreSQL port.
    postgres_db
        PostgreSQL database name.
    postgres_dsn
        DSN assembled from the individual postgres_* fields.
    ollama_host
        Base URL of the Ollama HTTP API.
    embed_model
        Ollama model used to produce embeddings for PubMed documents. Must
        produce 768-dimensional vectors (e.g. nomic-embed-text).
    guidelines_embed_model
        Ollama model used to embed Thuisarts guideline chunks. bge-m3 is
        multilingual, produces 1024-dimensional vectors, and has an 8192
        token context window — no truncation needed for triage sections.
    llm_model
        Ollama model used to generate answers from retrieved context.
        Overridable at runtime via --model or the LLM_MODEL env var.
    top_k
        Number of documents retrieved from pgvector and passed as context
        to the LLM. Higher values give more context at the cost of more
        tokens and slower inference.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_dsn(self) -> PostgresDsn:
        """DSN assembled from individual postgres_* fields."""
        dsn = f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        return PostgresDsn(dsn)

    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    guidelines_embed_model: str = "bge-m3"
    llm_model: str = "llama3.2:3b"
    top_k: int = 5
