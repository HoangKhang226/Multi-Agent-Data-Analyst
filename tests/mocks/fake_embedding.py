import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from typing import Any, List

class FakeEmbedding(BaseEmbedding):
    def __init__(self, dim=8, **kwargs: Any):
        super().__init__(embed_dim=dim, **kwargs)
        self.embed_dim = dim

    def _get_query_embedding(self, query: str) -> List[float]:
        np.random.seed(len(query))
        return np.random.rand(self.embed_dim).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        np.random.seed(len(text))
        return np.random.rand(self.embed_dim).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)