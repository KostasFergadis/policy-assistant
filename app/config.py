import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    ollama_url: str
    embedding_model: str
    answer_provider: str
    ollama_model: str
    ollama_think: bool
    claude_model: str
    min_similarity: float


def get_settings() -> Settings:
    load_dotenv()
    env = os.environ.get
    return Settings(
        database_url=env("DATABASE_URL", "postgresql://policy:policy@127.0.0.1:5433/policy"),
        ollama_url=env("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
        embedding_model=env("EMBEDDING_MODEL", "nomic-embed-text"),
        answer_provider=env("ANSWER_PROVIDER", "ollama").lower(),
        ollama_model=env("OLLAMA_MODEL", "qwen3:8b"),
        ollama_think=env("OLLAMA_THINK", "false").lower() == "true",
        claude_model=env("CLAUDE_MODEL", "claude-opus-5-5"),
        min_similarity=float(env("MIN_SIMILARITY", "0.5")),
    )
