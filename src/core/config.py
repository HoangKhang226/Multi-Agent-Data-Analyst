from pathlib import Path
import os
import logging
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from src.utils.logger import logger

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SETTING_FILE = _PROJECT_ROOT / "config" / "setting.yaml"


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini models."""

    model: str = "gemini-2.5-flash"
    summary_model: str = "gemini-2.5-flash"


class OllamaConfig(BaseModel):
    """Configuration for the locally hosted Ollama inference server."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    summary_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"

class RagConfig(BaseModel):
    """Configuration for RAG specific parameters."""
    reranker: str = "BAAI/bge-reranker-v2-m3"

class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model used in vector retrieval."""

    google: str
    ollama: str


class RetrievalConfig(BaseModel):
    """Retrieval parameters controlling similarity threshold and result count."""

    threshold: float
    top_k: int


class ChunkingConfig(BaseModel):
    """Chunking strategy parameters for hierarchical parent/child splitting."""

    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int


class PathConfig(BaseModel):
    """File system paths used by the application for data and storage."""

    data_dir: str
    data_csv_or_excel: str
    data_pdf_or_word: str
    vector_db: str
    collection_name: str


class MemoryConfig(BaseModel):
    """Configuration for Mem0 long-term user memory storage."""

    chroma_path: str = "storage/mem0_chroma"
    history_db_path: str = "storage/mem0_history.db"


class LangsmithConfig(BaseModel):
    """LangSmith tracing project settings."""

    project: str
    endpoint: str


class AppConfig(BaseModel):
    """Top-level application metadata."""

    project_name: str
    version: str
    debug: bool


class Settings(BaseSettings):
    """Application-wide settings loaded from .env and setting.yaml.

    API keys are read from environment variables; nested config blocks
    are populated from the YAML file passed to the constructor.
    """

    # API keys injected from environment / .env file
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")
    langsmith_api_key: str = Field(..., alias="LANGSMITH_API_KEY")

    app: AppConfig
    graph_provider: str
    memory_provider: str
    gemini: GeminiConfig = GeminiConfig()
    ollama: OllamaConfig
    rag: RagConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    storage: PathConfig
    memory: MemoryConfig = MemoryConfig()
    langsmith: LangsmithConfig

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


def load_setting() -> Settings:
    """Load and return application settings from the YAML config file.

    Reads setting.yaml, merges it with environment variables, and
    returns a validated Settings instance.

    Returns:
        A fully validated Settings object.

    Raises:
        FileNotFoundError: If setting.yaml does not exist.
    """
    if not _SETTING_FILE.exists():
        raise FileNotFoundError("setting.yaml not found")

    with open(_SETTING_FILE, "r", encoding="utf-8") as f:
        setting_config = yaml.safe_load(f)

    return Settings(**setting_config)


# Load settings at module level so other modules can import `settings` directly
try:
    settings = load_setting()
    logger.info("Settings loaded successfully")
    
    # Export ecosystem environment variables for automatic discovery by libraries
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

    # Export LangSmith environment variables for automatic tracing
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
        logger.info(f"LangSmith tracing enabled for project: {settings.langsmith.project}")

except Exception as e:
    logger.error(f"Error while loading settings: {e}")
    raise


if __name__ == "__main__":
    logger.info("Running main script")