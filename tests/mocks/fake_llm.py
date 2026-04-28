class FakeLLM:
    def invoke(self, prompt: str):
        prompt = prompt.lower()

        if "csv" in prompt or "data" in prompt:
            return "ROUTE: PANDAS"
        elif "document" in prompt or "pdf" in prompt:
            return "ROUTE: RAG"
        else:
            return "ROUTE: GENERAL"