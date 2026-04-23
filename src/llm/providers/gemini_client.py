from src.utils.logger import logger
from src.core.config import settings
from src.llm.base import BaseLLM
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
)


class GeminiClient(BaseLLM):
    """LangChain-compatible client for Google Gemini models.

    Wraps ChatGoogleGenerativeAI with project-level safety settings
    and exposes both plain and structured-output LLM variants.
    """

    def __init__(self, model_name: str, temperature: float = 0.0):
        """Initialize the Gemini client.

        Args:
            model_name: Gemini model identifier (e.g. "gemini-2.0-flash").
            temperature: Sampling temperature; 0.0 for deterministic output.
        """
        self.model_name = model_name
        self.temperature = temperature

        # Disable all content filters to avoid blocking legitimate document text
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        logger.debug(
            f"GeminiClient initialized — model: {self.model_name}, temperature: {self.temperature}"
        )

    def get_llm(self, **kwargs) -> ChatGoogleGenerativeAI:
        """Build and return a ChatGoogleGenerativeAI instance.

        Args:
            **kwargs: Additional keyword arguments forwarded to ChatGoogleGenerativeAI.

        Returns:
            A configured ChatGoogleGenerativeAI model ready for invocation.
        """
        logger.debug(f"Building Gemini LLM instance (model: {self.model_name})")
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=settings.google_api_key,
            safety_settings=self.safety_settings,
            **kwargs
        )

    def get_structed_llm(self, output_schema):
        """Return an LLM bound to a Pydantic schema for structured output.

        Args:
            output_schema: Pydantic model class the LLM response will be parsed into.

        Returns:
            A model instance that enforces the given output schema.
        """
        logger.debug(f"Building structured Gemini LLM for schema: {output_schema.__name__}")
        model = self.get_llm()
        return model.with_structured_output(output_schema)