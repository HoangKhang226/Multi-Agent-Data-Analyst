from src.core.config import settings
from src.utils.logger import logger
import json
from pathlib import Path
import shutil
import os
import gc

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


class VectorDBManager:
    def __init__(self, embedding_model, provider: str = "ollama"):
        self.embedding_model = embedding_model
        self.provider = provider.lower()
        
        # Handle provider aliases (ensure consistency with EmbeddingFactory)
        if self.provider == "gemini":
            self.provider = "google"

        # Set up base storage path
        self.base_storage_dir = Path(__file__).resolve().parents[2] / "storage"
        self.provider_dir = self.base_storage_dir / self.provider
        self.provider_dir.mkdir(parents=True, exist_ok=True)

        # Separate sub-directory for Qdrant segment files
        self._qdrant_path = self.provider_dir / "qdrant_storage"
        self._qdrant_path.mkdir(parents=True, exist_ok=True)

        self._db_client = None
        self._index = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def db_client(self) -> QdrantClient:
        """Lazy-initialise the local (embedded) QdrantClient."""
        if self._db_client is None:
            self._db_client = QdrantClient(path=str(self._qdrant_path))
        return self._db_client

    def _get_embedding_dimension(self) -> int:
        """
        Detect the vector dimension of the current embedding model.
        First tries a cheap attribute lookup; falls back to a test embedding.
        """
        try:
            if hasattr(self.embedding_model, "embed_dim"):
                return self.embedding_model.embed_dim
            sample = self.embedding_model.get_text_embedding("dimension test")
            return len(sample)
        except Exception as e:
            logger.warning(f"⚠️ Cannot detect embed_dim, defaulting to 768: {e}")
            return 768  # safe default for nomic-embed-text / bge-base

    def _collection_exists(self, collection_name: str) -> bool:
        """Check whether a Qdrant collection already exists."""
        existing = [c.name for c in self.db_client.get_collections().collections]
        return collection_name in existing

    def _collection_count(self, collection_name: str) -> int:
        """Return the number of points (vectors) in a Qdrant collection."""
        if not self._collection_exists(collection_name):
            return 0
        return self.db_client.count(collection_name=collection_name).count

    # ------------------------------------------------------------------
    # Path helpers (unchanged contract)
    # ------------------------------------------------------------------

    def get_persist_dir(self, collection_name: str) -> Path:
        """Return the collection-specific directory used for LlamaIndex metadata."""
        pdir = self.provider_dir / collection_name
        pdir.mkdir(parents=True, exist_ok=True)
        return pdir

    @property
    def summary_path(self) -> Path:
        """Path to the global summary JSON for this provider."""
        return self.provider_dir / "collection_metadata.json"

    # ------------------------------------------------------------------
    # Core storage context
    # ------------------------------------------------------------------

    def _get_storage_context(self, collection_name: str, persist_dir: Path = None) -> StorageContext:
        """
        Build a LlamaIndex StorageContext backed by a Qdrant vector store.

        If the Qdrant collection does not exist yet it is created with the
        correct vector dimension and cosine distance.
        """
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore

        # --- Ensure the Qdrant collection exists ---
        if not self._collection_exists(collection_name):
            embed_dim = self._get_embedding_dimension()
            self.db_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            logger.info(f"✅ Created Qdrant collection '{collection_name}' (dim={embed_dim})")

        vector_store = QdrantVectorStore(
            client=self.db_client,
            collection_name=collection_name,
        )

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
            index_store=SimpleIndexStore(),
        )

    # ------------------------------------------------------------------
    # DB lifecycle
    # ------------------------------------------------------------------

    def reset_db(self):
        """
        Completely wipe all Qdrant data and LlamaIndex metadata for this
        provider.  Useful when the embedding model (and therefore vector
        dimensions) changes.
        """
        try:
            # Close the Qdrant client before deleting its files
            if self._db_client is not None:
                self._db_client.close()

            self._index = None
            self._db_client = None
            gc.collect()

            if self.provider_dir.exists():
                shutil.rmtree(self.provider_dir, ignore_errors=True)
                logger.info(f"🧹 Cleaned up storage directory: {self.provider_dir}")

            # Re-create expected directory structure
            self.provider_dir.mkdir(parents=True, exist_ok=True)
            self._qdrant_path = self.provider_dir / "qdrant_storage"
            self._qdrant_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reset DB: {e}")
            return False

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def get_index(self, collection_name: str = "default_collection"):
        """Load and return the existing VectorStoreIndex, or None if absent."""
        if self._index is not None:
            return self._index

        try:
            primary_dir = self.get_persist_dir(collection_name)
            legacy_dir = self.provider_dir

            # Determine which metadata directory to load from
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
                # No docstore found anywhere — no index yet
                return None

            # --- Legacy collection fallback (handles renamed collections) ---
            provider_suffix = f"_{self.provider}"
            if not collection_name.endswith(provider_suffix):
                if self._collection_count(collection_name) == 0:
                    legacy_name = f"{collection_name}{provider_suffix}"
                    if self._collection_count(legacy_name) > 0:
                        logger.info(
                            f"⚠️ Primary collection '{collection_name}' is empty. "
                            f"Falling back to legacy '{legacy_name}'"
                        )
                        storage_context = self._get_storage_context(
                            legacy_name, persist_dir=target_dir
                        )
                        index = load_index_from_storage(
                            storage_context,
                            embed_model=self.embedding_model,
                        )

            self._index = index
            return self._index

        except Exception as e:
            logger.warning(f"⚠️ Error loading index for '{collection_name}': {e}")
            return None

    def add_documents(self, nodes, collection_name: str = "default_collection"):
        """
        Insert nodes into the index.  Skips nodes that already exist to
        avoid duplicates.  Persists metadata (docstore, index_store) after
        every call.
        """
        try:
            # 1. Deduplicate input nodes
            unique_nodes_dict = {n.node_id: n for n in nodes}
            unique_input_nodes = list(unique_nodes_dict.values())

            # 2. Load or create index
            self._index = self.get_index(collection_name)

            if self._index is None:
                logger.info("🚀 No existing index found — creating a new one...")
                storage_context = self._get_storage_context(collection_name)
                self._index = VectorStoreIndex(
                    nodes=unique_input_nodes,
                    storage_context=storage_context,
                    embed_model=self.embedding_model,
                    show_progress=True,
                    store_nodes_override=True,  # Required for hierarchical RAG
                )
            else:
                logger.info("➕ Existing index found — checking for new nodes...")
                existing_ids = set(self._index.storage_context.docstore.docs.keys())
                final_new_nodes = [n for n in unique_input_nodes if n.node_id not in existing_ids]

                if final_new_nodes:
                    logger.info(f"Inserting {len(final_new_nodes)} new nodes.")
                    self._index.insert_nodes(final_new_nodes)
                else:
                    logger.info("ℹ️ All nodes already present in the index.")

            # 3. Persist LlamaIndex metadata (docstore + index_store)
            persist_dir = self.get_persist_dir(collection_name)
            self._index.storage_context.persist(persist_dir=str(persist_dir))
            logger.info(f"✅ Persisted index metadata for '{collection_name}' → {persist_dir}")

            return [n.node_id for n in unique_input_nodes]

        except Exception as e:
            logger.error(f"❌ Error in add_documents: {e}")
            raise

    # ------------------------------------------------------------------
    # Retrievers
    # ------------------------------------------------------------------

    def get_retriever(
        self,
        similarity_top_k: int = 3,
        collection_name: str = "default_collection",
        **kwargs,
    ):
        """
        Return an AutoMergingRetriever (hierarchical RAG).

        Leaf nodes are retrieved via vector similarity; the retriever then
        automatically merges them into their parent chunks when a threshold
        is met, returning richer context to the LLM.
        """
        index = self.get_index(collection_name)
        if index is None:
            logger.error("❌ Cannot build retriever — index is empty.")
            return None

        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k, **kwargs)

        logger.info("🔍 Initialising AutoMergingRetriever (hierarchical)...")
        retriever = AutoMergingRetriever(
            vector_retriever,
            index.storage_context,
            verbose=True,
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
        Return a Hybrid Retriever that fuses:
          - Vector search (semantic similarity via Qdrant)
          - BM25 (keyword / exact-match search)

        Fusion is performed with Reciprocal Rank Fusion (RRF) via
        QueryFusionRetriever.

        Args:
            similarity_top_k: Final number of results after fusion.
            collection_name:  Target Qdrant collection.
            num_queries:      Number of synthetic sub-queries generated by
                              the LLM (1 = use the original query only).
        """
        index = self.get_index(collection_name)
        if index is None:
            logger.error("❌ Cannot build hybrid retriever — index is empty.")
            return None

        # BM25 requires in-memory node list from the docstore
        all_nodes = list(index.storage_context.docstore.docs.values())

        if not all_nodes:
            logger.warning("⚠️ Docstore is empty — BM25 has no data. Falling back to vector retriever.")
            return self.get_retriever(similarity_top_k=similarity_top_k,
                                      collection_name=collection_name, **kwargs)

        # --- Vector retriever (semantic) ---
        vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k, **kwargs)

        # --- BM25 retriever (keyword) ---
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=similarity_top_k,
        )

        # --- Fuse both with RRF ---
        logger.info("🔀 Initialising Hybrid Retriever (BM25 + Vector / RRF)...")
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=similarity_top_k,
            num_queries=num_queries,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True,
        )
        return hybrid_retriever

    # ------------------------------------------------------------------
    # Summary helpers (unchanged)
    # ------------------------------------------------------------------

    def save_summary(self, collection_name: str, summary: str):
        """Append or update a document summary in the metadata JSON store."""
        try:
            data = self.get_summary()
            data[collection_name] = summary
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ Summary saved for '{collection_name}' → {self.summary_path}")
        except Exception as e:
            logger.error(f"❌ Error saving summary: {e}")

    def get_summary(self, collection_name: str = None) -> dict:
        """
        Retrieve summary information.
        - With collection_name: return the summary string for that collection.
        - Without collection_name: return the full metadata dictionary.
        """
        if not hasattr(self, "summary_path") or not self.summary_path.exists():
            return {}

        try:
            if self.summary_path.stat().st_size == 0:
                return {}

            with open(self.summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if collection_name:
                summary = data.get(collection_name)
                if not summary:
                    key = f"{collection_name}_{self.provider}"
                    summary = data.get(key, {})
                return summary

            return data

        except json.JSONDecodeError:
            logger.error(f"❌ Metadata file corrupted (JSON error): {self.summary_path}")
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error reading summary: {e}")
            return {}