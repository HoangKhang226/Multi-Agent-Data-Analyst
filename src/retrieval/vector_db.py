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

        # Set up base storage path
        self.base_storage_dir = Path(__file__).resolve().parents[2] / "storage"
        self.provider_dir = self.base_storage_dir / self.provider
        self.provider_dir.mkdir(parents=True, exist_ok=True)
        
        self._db_client = None
        self._index = None

    @property
    def db_client(self):
        """Initialize and return the ChromaDB client on demand."""
        if self._db_client is None:
            # ChromaDB stays at provider level (shares same sqlite normally)
            self._db_client = chromadb.PersistentClient(path=str(self.provider_dir))
        return self._db_client

    def get_persist_dir(self, collection_name: str) -> Path:
        """Get collection-specific persist directory for LlamaIndex metadata."""
        pdir = self.provider_dir / collection_name
        pdir.mkdir(parents=True, exist_ok=True)
        return pdir

    @property
    def summary_path(self) -> Path:
        """Global metadata for summaries in this provider folder."""
        return self.provider_dir / "collection_metadata.json"

    def _get_storage_context(self, collection_name: str, persist_dir: Path = None):
        """Get the LlamaIndex StorageContext for a specific collection and path."""
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        
        chroma_collection = self.db_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        if persist_dir is None:
            persist_dir = self.get_persist_dir(collection_name)
            
        docstore_path = persist_dir / "docstore.json"
        
        if docstore_path.exists():
            try:
                return StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(persist_dir),
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load storage context from {persist_dir}: {e}")
        
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

            if self.provider_dir.exists():
                shutil.rmtree(self.provider_dir,
                ignore_errors=True # ignore errors if file is locked
                )
                logger.info(f"🧹 Cleaned up storage directory: {self.provider_dir}")
            
            self.provider_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reset DB: {e}")
            return False

    def get_index(self, collection_name: str = "default_collection"):
        """Load and return the existing index if available."""
        if self._index is not None:
            return self._index

        try:
            # Multi-path search for metadata
            primary_dir = self.get_persist_dir(collection_name)
            legacy_dir = self.provider_dir
            
            # Decide which directory to use for LlamaIndex files
            target_dir = None
            if (primary_dir / "docstore.json").exists():
                target_dir = primary_dir
            elif (legacy_dir / "docstore.json").exists():
                target_dir = legacy_dir
                logger.info(f"ℹ️ Falling back to legacy root metadata for '{collection_name}'")

            if target_dir:
                storage_context = self._get_storage_context(collection_name, persist_dir=target_dir)
                index = load_index_from_storage(
                    storage_context,
                    embed_model=self.embedding_model,
                )
            else:
                # If no docstore exists anywhere, index cannot be loaded
                return None
            
            # --- Fallback Logic for Content ---
            # If the loaded index is empty (0 nodes in Chroma for this collection name),
            # try to find a legacy collection name (e.g. chat_with_data_gemini instead of chat_with_data)
            provider_suffix = f"_{self.provider}"
            if not collection_name.endswith(provider_suffix):
                # We need to check the actual node count in the vector store
                chroma_collection = self.db_client.get_collection(collection_name)
                if chroma_collection.count() == 0:
                    legacy_name = f"{collection_name}{provider_suffix}"
                    try:
                        legacy_chroma = self.db_client.get_collection(legacy_name)
                        if legacy_chroma.count() > 0:
                            logger.info(f"⚠️ Primary collection '{collection_name}' is empty. Falling back to legacy name '{legacy_name}'")
                            # We reload the index using the legacy collection name but same metadata dir
                            storage_context = self._get_storage_context(legacy_name, persist_dir=target_dir)
                            index = load_index_from_storage(
                                storage_context,
                                embed_model=self.embedding_model,
                            )
                    except Exception: pass

            self._index = index
            return self._index
        except Exception as e:
            logger.warning(f"⚠️ Error loading index for {collection_name}: {e}")
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
                ) # Create new index from nodes
            else:
                logger.info("➕ Existing index found, checking for new nodes...")

                existing_ids = set(
                    self._index.storage_context.docstore.docs.keys()
                ) # Get list of existing node IDs
 
                final_new_nodes = [
                    n for n in unique_input_nodes if n.node_id not in existing_ids
                ] # Filter for truly new nodes

                if final_new_nodes:
                    logger.info(f"Inserting {len(final_new_nodes)} new nodes into the DB.")
                    self._index.insert_nodes(final_new_nodes) # Add new nodes to index
                else:
                    logger.info("ℹ️ All nodes already exist in the index.")

            # Persist changes
            persist_dir = self.get_persist_dir(collection_name)
            self._index.storage_context.persist(
                persist_dir=str(persist_dir)
            )
            logger.info(f"✅ Successfully persisted state for '{collection_name}' at {persist_dir}")

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
    # hierachical retriever
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
            index.storage_context, # storage context containing all LlamaIndex components
            verbose=True
        ) # Retriever capable of merging small chunks into larger parent context

        return retriever

    def get_hybrid_retriever(
        self,
        similarity_top_k: int = 3,
        collection_name: str = "default_collection",
        num_queries: int = 1,
        **kwargs,
    ):
    # hybrid retriever
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

        # Lấy tất cả nodes từ docstore để build BM25 index
        all_nodes = list(index.storage_context.docstore.docs.values())

        if not all_nodes:
            logger.warning("⚠️ Docstore trống, BM25 không có dữ liệu để index.")
            return self.get_retriever(similarity_top_k=similarity_top_k,
                                     collection_name=collection_name, **kwargs)

        # Vector retriever (semantic search)
        vector_retriever = index.as_retriever(
            similarity_top_k=similarity_top_k, **kwargs
        )

        # BM25 retriever (keyword search)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=similarity_top_k,
        )

        # Kết hợp cả hai bằng QueryFusionRetriever (RRF)
        logger.info("🔀 Đang khởi tạo Hybrid Retriever (BM25 + Vector Search)...")
        # Hybrid Search (BM25 + Vector Search)
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
            data = self.get_summary() # Fetch entire map
            # We save under both keys to ensure compatibility during transition
            data[collection_name] = summary
            
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"✅ Summary saved successfully for '{collection_name}' at {self.summary_path}")

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
                # Try direct name first, then suffixed legacy name
                summary = data.get(collection_name)
                if not summary:
                    key = f"{collection_name}_{self.provider}"
                    summary = data.get(key, {})
                return summary
            
            return data

        except json.JSONDecodeError:
            logger.error(f"❌ Metadata file is corrupted (JSON error) at {self.summary_path}")
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error while reading summary: {e}")
            return {}