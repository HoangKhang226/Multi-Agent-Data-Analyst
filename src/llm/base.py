from pydantic import BaseModel
from abc import abstractmethod, ABC
from typing import Type, TypeVar, Any

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    """Abstract base class that all LLM provider clients must implement.

    Subclasses are responsible for returning a configured LLM instance
    for plain chat completions and for structured (schema-bound) outputs.
    """

    @abstractmethod
    def get_llm(self, **kwargs) -> Any:
        """Return a plain chat LLM instance.

        Args:
            **kwargs: Provider-specific keyword arguments forwarded to the client.

        Returns:
            A chat model object ready for invocation.
        """
        pass

    @abstractmethod
    def get_structed_llm(self, output_schema: Type[T]) -> Any:
        """Return an LLM bound to a Pydantic output schema for structured output.

        Args:
            output_schema: A Pydantic BaseModel subclass defining the expected output.

        Returns:
            A model instance that parses responses into the given schema.
        """
        pass