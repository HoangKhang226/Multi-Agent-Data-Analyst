from typing import Any, Type

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.core.config import settings
from src.llm.base import BaseLLM
from src.utils.logger import logger


class OllamaClient(BaseLLM):
    """LangChain-compatible client for locally hosted Ollama models.

    Wraps ``ChatOllama`` and exposes both plain and structured-output
    LLM variants, mirroring the interface of :class:`GeminiClient`.
    """

    def __init__(self, model_name: str, temperature: float = 0.0):
        """Initialise the Ollama client.

        Args:
            model_name:  Ollama model tag pulled locally (e.g. ``"qwen3:8b"``).
            temperature: Sampling temperature; 0.0 for deterministic output.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = settings.ollama.base_url

        logger.debug(
            f"OllamaClient initialized — model: {self.model_name}, "
            f"temperature: {self.temperature}, base_url: {self.base_url}"
        )

    def get_llm(self, **kwargs) -> ChatOllama:
        """Build and return a ``ChatOllama`` instance.

        Args:
            **kwargs: Additional keyword arguments forwarded to ``ChatOllama``.

        Returns:
            A configured ``ChatOllama`` model ready for invocation.
        """
        logger.debug(f"Building Ollama LLM instance (model: {self.model_name})")
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url,
            **kwargs,
        )

    def get_structed_llm(self, output_schema: Type[BaseModel]) -> Any:
        """Return an LLM bound to a Pydantic schema for structured output.

        Uses LangChain's ``.with_structured_output()`` which falls back to
        JSON-mode prompting for models that do not natively support tool-call
        based structured output.

        Args:
            output_schema: Pydantic model class the LLM response is parsed into.

        Returns:
            A model instance that enforces the given output schema.
        """
        logger.debug(
            f"Building structured Ollama LLM for schema: {output_schema.__name__}"
        )
        model = self.get_llm()
        return model.with_structured_output(output_schema)