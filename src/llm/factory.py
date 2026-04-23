from typing import Literal, Union

from src.core.config import settings
from src.llm.base import BaseLLM
from src.llm.providers.gemini_client import GeminiClient
from src.llm.providers.ollama_client import OllamaClient
from src.utils.logger import logger

# LlamaIndex Global Settings
from llama_index.core import Settings


class LLMFactory:
    """Factory for creating pre-configured LLM clients by purpose.

    Selects the provider (Gemini or Ollama) from ``settings.graph_provider``
    and picks the appropriate model name + temperature for each purpose.

    Supported purposes
    ------------------
    - ``"summary"``    — creative generation, uses ``summary_model`` + configured temperature
    - ``"rag"``        — deterministic retrieval tasks, forces ``temperature=0.0``
    - ``"classifier"`` — deterministic classification, forces ``temperature=0.0``
    """

    @staticmethod
    def create_client(
        purpose: Literal["summary", "rag", "classifier"],
        provider: str = None,
    ) -> Union[GeminiClient, OllamaClient]:
        """Return an LLM client configured for the given purpose.

        Args:
            purpose: Intended use case — ``"summary"``, ``"rag"``, or ``"classifier"``.
            provider: Optional provider override (``"gemini"`` or ``"ollama"``).
                     If None, defaults to ``settings.graph_provider``.

        Returns:
            A :class:`GeminiClient` or :class:`OllamaClient` with the
            model and temperature appropriate for *purpose*.

        Raises:
            ValueError: If an unsupported purpose string is provided.
            ValueError: If the provider is not ``"gemini"`` or ``"ollama"``.
        """
        if provider is None:
            provider = settings.graph_provider.lower()
        else:
            provider = provider.lower()

        # Handle provider aliases
        if provider == "google":
            provider = "gemini"

        if purpose == "summary":
            # Default temperature for summary tasks
            temperature = 0.7 
            if provider == "gemini":
                model = settings.gemini.summary_model
                logger.debug(
                    f"Creating Gemini LLM for SUMMARY (model: {model}, temp: {temperature})"
                )
                return GeminiClient(model_name=model, temperature=temperature)
            elif provider == "ollama":
                model = settings.ollama.summary_model
                logger.debug(
                    f"Creating Ollama LLM for SUMMARY (model: {model}, temp: {temperature})"
                )
                return OllamaClient(model_name=model, temperature=temperature)
            else:
                raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'gemini' or 'ollama'.")

        elif purpose in ("rag", "classifier"):
            # Both tasks require deterministic output — force temperature to 0
            temperature = 0.0
            if provider == "gemini":
                model = settings.gemini.model
                logger.debug(
                    f"Creating Gemini LLM for {purpose.upper()} (model: {model}, temp: {temperature})"
                )
                return GeminiClient(model_name=model, temperature=temperature)
            elif provider == "ollama":
                model = settings.ollama.model
                logger.debug(
                    f"Creating Ollama LLM for {purpose.upper()} (model: {model}, temp: {temperature})"
                )
                return OllamaClient(model_name=model, temperature=temperature)
            else:
                raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'gemini' or 'ollama'.")

        else:
            logger.error(f"Unsupported LLM purpose: '{purpose}'")
            raise ValueError(f"LLM purpose not supported: {purpose}")

    @staticmethod
    def configure_llama_index_settings(provider: str = None):
        """Configure LlamaIndex global Settings to use the chosen provider.
        
        This prevents LlamaIndex from defaulting to OpenAI for internal tasks.
        """
        if provider is None:
            provider = settings.graph_provider.lower()
        else:
            provider = provider.lower()
            
        if provider == "google":
            provider = "gemini"
            
        logger.info(f"Configuring LlamaIndex global Settings for: {provider}")
        
        if provider == "gemini":
            from llama_index.llms.google_genai import GoogleGenAI
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
            
            Settings.llm = GoogleGenAI(
                model=settings.gemini.model,
                api_key=settings.google_api_key
            )
            Settings.embed_model = GoogleGenAIEmbedding(
                model_name=settings.embedding.google,
                api_key=settings.google_api_key
            )
            
        elif provider == "ollama":
            from llama_index.llms.ollama import Ollama
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            Settings.llm = Ollama(
                model=settings.ollama.model,
                base_url=settings.ollama.base_url
            )
            Settings.embed_model = OllamaEmbedding(
                model_name=settings.embedding.ollama,
                base_url=settings.ollama.base_url
            )
        
        logger.info(f"LlamaIndex global Settings updated successfully for {provider}")


def get_llm_provider(purpose: str = "rag", provider: str = None):
    """Convenience helper to get a pre-configured LLM client instance."""
    return LLMFactory.create_client(purpose=purpose, provider=provider).get_llm()

