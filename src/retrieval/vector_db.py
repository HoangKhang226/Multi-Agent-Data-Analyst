from src.core.config import settings
from src.utils.logger import logger
import json
from pathlib import Path
import chromadb
import shutil
import os
import gc
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

class VectorDBManager:
    def __init__(self, embedding_model, provider: str = "ollama"):
        self.embedding_model = embedding_model
        self.provider = provider.lower()

        # Set up storage paths
        base_dir = Path(__file__).resolve().parents[2] / "storage"
        self.persist_directory = base_dir / self.provider
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.summary_path = self.persist_directory / "collection_metadata.json"
        self._db_client = None
        self._index = None

    @property
    def db_client(self):
        """Initialize and return the ChromaDB client on demand."""
        if self._db_client is None:
            self._db_client = chromadb.PersistentClient(path=str(self.persist_directory)) # initial persistent memory
        return self._db_client

    def _get_storage_context(self, collection_name: str):
        """Get the LlamaIndex StorageContext for a specific collection."""
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        
        chroma_collection = self.db_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Check if we have existing storage files
        docstore_path = self.persist_directory / "docstore.json"
        
        if docstore_path.exists():
            try:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(self.persist_directory),
                )
                return storage_context
            except Exception as e:
                logger.warning(f"⚠️ Failed to load storage context from {self.persist_directory}: {e}")
        
        # Initialize new storage context with explicit stores
        return StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore()
        )

    def reset_db(self):
        """Completely reset the database to resolve dimension mismatch issues."""
        try:
            # 1. Close connections and release objects
            self._index = None
            self._db_client = None
            
            gc.collect() # Force garbage collection to release file locks

            if self.persist_directory.exists():
                shutil.rmtree(self.persist_directory,
                ignore_errors=True # nếu file không xóa được sẽ bỏ qua
                )
                logger.info(f"🧹 Cleaned up storage directory: {self.persist_directory}")
            
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reset DB: {e}")
            return False

    def get_index(self, collection_name: str = "default_collection"):
        """Load and return the existing index if available."""
        if self._index is not None:
            return self._index

        if not (self.persist_directory / "docstore.json").exists():
            return None

        try:
            storage_context = self._get_storage_context(collection_name)
            self._index = load_index_from_storage(
                storage_context,
                embed_model=self.embedding_model,
            )
            return self._index
        except Exception:
            return None

    def add_documents(self, nodes, collection_name: str = "default_collection"):
        """Add nodes to the index and persist changes."""
        try:
            # 1. Deduplicate nodes
            unique_nodes_dict = {n.node_id: n for n in nodes}
            unique_input_nodes = list(unique_nodes_dict.values())

            # 2. Load or create index
            self._index = self.get_index(collection_name)

            if self._index is None:
                logger.info("🚀 Index doesn't exist or was reset, initializing new index...")
                storage_context = self._get_storage_context(collection_name)

                self._index = VectorStoreIndex(
                    nodes=unique_input_nodes,
                    storage_context=storage_context,
                    embed_model=self.embedding_model,
                    show_progress=True,
                    store_nodes_override=True, # Critical for hierarchical RAG persistence
                )
            else:
                logger.info("➕ Existing index found, checking for new nodes...")

                existing_ids = set(
                    self._index.storage_context.docstore.docs.keys()
                )

                final_new_nodes = [
                    n for n in unique_input_nodes if n.node_id not in existing_ids
                ]

                if final_new_nodes:
                    logger.info(f"Inserting {len(final_new_nodes)} new nodes into the DB.")
                    self._index.insert_nodes(final_new_nodes)
                else:
                    logger.info("ℹ️ All nodes already exist in the index.")

            # 3. Persist changes
            self._index.storage_context.persist(
                persist_dir=str(self.persist_directory)
            )
            logger.info(f"✅ Successfully persisted state at {self.persist_directory}")

            return [n.node_id for n in unique_input_nodes]

        except Exception as e:
            logger.error(f"❌ Error in add_documents: {e}")
            raise e

    def get_retriever(
        self,
        similarity_top_k: int = 3,
        collection_name: str = "default_collection",
        **kwargs,
    ):
        """Return an AutoMergingRetriever that automatically merges retrieved child nodes into parents."""
        index = self.get_index(collection_name)

        if index is None:
            logger.error("❌ Cannot obtain retriever because the index is empty.")
            return None

        # 1. Base retriever for leaf node matching
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k, **kwargs)

        # 2. Use AutoMergingRetriever with the existing storage context
        # This will automatically handle the mapping using the docstore in storage_context
        logger.info("Initializing AutoMergingRetriever...")
        retriever = AutoMergingRetriever(
            vector_retriever,
            index.storage_context,
            verbose=True
        )

        return retriever

    def get_hybrid_retriever(
        self,
        similarity_top_k: int = 3,
        collection_name: str = "default_collection",
        num_queries: int = 1,
        **kwargs,
    ):
        """
        Trả về Hybrid Retriever kết hợp:
          - Vector Search (semantic): tìm theo ngữ nghĩa
          - BM25 (keyword):          tìm theo từ khoá chính xác
        Hai retriever được hợp nhất bằng QueryFusionRetriever
        với thuật toán Reciprocal Rank Fusion (RRF).

        Tham số:
            similarity_top_k: Số kết quả trả về sau khi fusion
            collection_name:   Tên collection trong ChromaDB
            num_queries:       Số câu query phụ được sinh ra (1 = chỉ dùng câu gốc,
                               tăng lên để tự động sinh thêm câu query đa dạng hơn)
        """
        index = self.get_index(collection_name)

        if index is None:
            logger.error("❌ Không thể tạo hybrid retriever vì index trống.")
            return None

        # --- 1. Lấy tất cả nodes từ docstore để build BM25 index ---
        all_nodes = list(index.storage_context.docstore.docs.values())

        if not all_nodes:
            logger.warning("⚠️ Docstore trống, BM25 không có dữ liệu để index.")
            return self.get_retriever(similarity_top_k=similarity_top_k,
                                     collection_name=collection_name, **kwargs)

        # --- 2. Vector retriever (semantic search) ---
        vector_retriever = index.as_retriever(
            similarity_top_k=similarity_top_k, **kwargs
        )

        # --- 3. BM25 retriever (keyword search) ---
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=similarity_top_k,
        )

        # --- 4. Kết hợp cả hai bằng QueryFusionRetriever (RRF) ---
        logger.info("🔀 Đang khởi tạo Hybrid Retriever (BM25 + Vector Search)...")
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,       # 1 = không sinh query phụ, chỉ dùng câu gốc
            mode="reciprocal_rerank",       # Thuật toán RRF – phổ biến nhất cho hybrid search
            use_async=False,
            verbose=True,
        )

        return hybrid_retriever

    def save_summary(self, collection_name: str, summary: str):
        """Append or update a document summary in the metadata storage."""
        try:
            data = self.get_summary() # Fetch existing metadata map
            key = f"{collection_name}_{self.provider}"
            data[key] = summary

            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"✅ Summary saved successfully to {self.summary_path}")

        except Exception as e:
            logger.error(f"❌ Error while saving summary: {e}")

    def get_summary(self, collection_name: str = None) -> dict:
        """
        Retrieve summary information.
        - If collection_name is provided: Return summary string/obj for that collection.
        - Otherwise: Return the entire metadata dictionary.
        """
        if not hasattr(self, 'summary_path') or not self.summary_path.exists():
            return {}

        try:
            if self.summary_path.stat().st_size == 0:
                return {}

            with open(self.summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if collection_name:
                key = f"{collection_name}_{self.provider}"
                return data.get(key, {})
            
            return data

        except json.JSONDecodeError:
            logger.error(f"❌ Metadata file is corrupted (JSON error) at {self.summary_path}")
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error while reading summary: {e}")
            return {}