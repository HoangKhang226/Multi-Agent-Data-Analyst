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
        """Tìm kiếm thuần Vector Search (semantic). Dùng AutoMergingRetriever."""
        logger.info(f"[Vector Search] Đang tìm kiếm: {hyde}")
        retriever = self.vector_db.get_retriever(collection_name=collection_name, similarity_top_k=k)
        try:
            docs = retriever.retrieve(hyde)
            logger.info(f"Tìm thấy {len(docs)} tài liệu liên quan.")
            return docs
        except Exception as e:
            logger.error(f"Lỗi khi truy xuất tài liệu: {e}")
            return []

    def retrieval_hybrid(
        self,
        query: str,
        collection_name: str = "default_collection",
        k: int = settings.retrieval.top_k,
        num_queries: int = 1,
    ):
        """
        Tìm kiếm Hybrid = BM25 (từ khoá) + Vector Search (ngữ nghĩa),
        kết hợp bằng thuật toán Reciprocal Rank Fusion (RRF).

        Tham số:
            query:        Câu hỏi của người dùng (có thể là HyDE hoặc câu gốc)
            collection_name: Tên collection trong ChromaDB
            k:            Số kết quả tối đa trả về
            num_queries:  Số câu query phụ (1 = chỉ dùng câu gốc, không tốn LLM call)
        """
        logger.info(f"[Hybrid Search] Đang tìm kiếm: {query}")
        retriever = self.vector_db.get_hybrid_retriever(
            similarity_top_k=k,
            collection_name=collection_name,
            num_queries=num_queries,
        )

        if retriever is None:
            logger.warning("⚠️ Hybrid retriever không khởi tạo được, fallback sang Vector Search.")
            return self.retrieval(query, collection_name=collection_name, k=k)

        try:
            docs = retriever.retrieve(query)
            logger.info(f"[Hybrid Search] Tìm thấy {len(docs)} tài liệu sau khi fusion.")
            return docs
        except Exception as e:
            logger.error(f"❌ Lỗi khi truy xuất hybrid: {e}")
            return []
