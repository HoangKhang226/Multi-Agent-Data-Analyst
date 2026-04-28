from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class FakeLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = prompt.lower()

        if "csv" in prompt or "data" in prompt:
            return "ROUTE: PANDAS"
        elif "document" in prompt or "pdf" in prompt:
            return "ROUTE: RAG"
        else:
            return "ROUTE: GENERAL"

    @property
    def _llm_type(self) -> str:
        return "fake"