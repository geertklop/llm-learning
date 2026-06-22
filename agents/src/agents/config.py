"""Application configuration for the agents project."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to this file so it works regardless of working directory.
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or a .env file.

    Attributes
    ----------
    ollama_host
        Base URL of the Ollama HTTP API.
    ollama_model
        Ollama model used for the agent LLM. Must support tool calling
        (e.g. llama3.1:8b, llama3.2:3b, qwen2.5:7b).
    ollama_embed_model
        Ollama model used for producing embeddings. Must match the model used
        during RAG ingestion (e.g. bge-m3 for guidelines).
    postgres_host
        Hostname of the PostgreSQL server.
    postgres_port
        Port the PostgreSQL server listens on.
    postgres_user
        PostgreSQL user to authenticate as.
    postgres_password
        Password for the PostgreSQL user.
    postgres_db
        PostgreSQL database name.
    """

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embed_model: str = "bge-m3"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "llm"
    postgres_password: str = "llm"
    postgres_db: str = "llm"
