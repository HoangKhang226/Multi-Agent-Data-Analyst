from src.core.config import settings
from src.utils.logger import logger
from src.retrieval.vector_db import VectorDBManager
from src.llm.embeddings import EmbeddingFactory


class Retriever():
    def __init__(self, provider: str = "ollama"):
        embed_factory = EmbeddingFactory()
        self.embedding_model = embed_factory.get_embedding(provider=provider)

        self.vector_db = VectorDBManager(embedding_model=self.embedding_model, provider=provider)
    
        logger.info(f"Retriever đã tạo với provider: {provider}")

    def retrieval(
        self,
        hyde: str,
        collection_name: str = "default_collection",
        k: int = settings.retrieval.top_k,
    ):
        logger.info(f"đang tìm kiếm bằng hyde: {hyde}")
        retriever = self.vector_db.get_retriever(collection_name=collection_name, similarity_top_k=k)
        try:
            docs = retriever.retrieve(hyde)
            logger.info(f"Tìm thấy {len(docs)} tài liệu liên quan.")
            return docs
        except Exception as e:
            logger.error(f"Lỗi khi truy xuất tài liệu: {e}")
            return []