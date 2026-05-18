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
    """

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
