from src.core.config import settings
from src.utils.logger import logger
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from typing import Union


class EmbeddingFactory():
    def __init__(self):
        self.google_model = settings.embedding.google
        self.ollama_model = settings.embedding.ollama
        self.ollama_base_url = settings.ollama.base_url
        self.google_api_key = settings.google_api_key

    def get_embedding(
        self, provider: str = None
    ) -> Union[GoogleGenAIEmbedding, OllamaEmbedding]:
        """Return a LlamaIndex-compatible embedding model instance for the given provider.

        Args:
            provider: "google" or "ollama". Defaults to settings.llm.provider.
        """
        if provider is None:
            # Fallback to general LLM provider if not specified
            provider = settings.embedding.provider.lower()
        else:
            provider = provider.lower()

        # Handle provider aliases
        if provider == "gemini":
            provider = "google"

        if provider == "ollama":
            logger.debug(f"Initializing Ollama embeddings (model: {self.ollama_model})")
            return OllamaEmbedding(
                model_name=self.ollama_model, 
                base_url=self.ollama_base_url
            )
        elif provider == "google":
            logger.debug(f"Initializing Google embeddings (model: {self.google_model})")
            return GoogleGenAIEmbedding(
                model_name=f"models/{self.google_model}", 
                api_key=self.google_api_key
            )
        else:
            raise ValueError(
                f"Provider {provider} not supported. Use 'google' or 'ollama'."
            )
