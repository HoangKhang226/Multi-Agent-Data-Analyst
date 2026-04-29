# `src/retrieval/` — Vector Search & Retrieval Engine

This module implements the document retrieval layer of the RAG pipeline. It abstracts over [Qdrant](https://qdrant.tech/) (vector database) and [LlamaIndex](https://www.llamaindex.ai/) to provide three retrieval strategies: **Hierarchical (AutoMerging)**, **Hybrid (BM25 + Vector)**, and is designed to work with **HyDE** queries from the agent layer.

---

## File Overview

| File | Responsibility |
|---|---|
| `vector_db.py` | `VectorDBManager` — low-level interface to Qdrant + LlamaIndex (indexing, retrieval, persistence) |
| `engine.py` | `Retriever` — high-level facade that selects retrieval mode and delegates to `VectorDBManager` |

---

## Architecture

```
Agent (hyde node)
      │
      │  hyde_query (hypothetical document)
      ▼
  Retriever.retrieval(hyde_query, mode)
      │
      ├─ mode="hierarchical" ──► VectorDBManager.get_retriever()
      │                              └─► AutoMergingRetriever (LlamaIndex)
      │                                    └─► Qdrant (cosine vector search)
      │                                          └─► merge leaf → parent chunks
      │
      └─ mode="hybrid" ────────► VectorDBManager.get_hybrid_retriever()
                                      └─► QueryFusionRetriever (LlamaIndex)
                                            ├─► Qdrant vector retriever (semantic)
                                            └─► BM25Retriever (keyword/exact match)
                                                  (fused via Reciprocal Rank Fusion)
```

---

## Retrieval Strategies

### 1. Hierarchical RAG (`mode="hierarchical"`) — *Default*

Uses LlamaIndex's `AutoMergingRetriever`. Documents are indexed as a **parent-child node tree** (large parent chunks → small leaf chunks). At query time:

1. Vector similarity search finds the most relevant **leaf nodes**.
2. If enough sibling leaves from the same parent are retrieved, they are **merged** into the parent chunk.
3. The LLM receives richer, more coherent context.

**Best for**: long documents, books, reports where local context alone is insufficient.

### 2. Hybrid Search (`mode="hybrid"`)

Combines two complementary signals via **Reciprocal Rank Fusion (RRF)**:

| Retriever | Signal type | Strengths |
|---|---|---|
| Qdrant (dense) | Semantic similarity | Handles paraphrasing, synonyms |
| BM25 | Keyword / TF-IDF | Handles exact terms, codes, names |

The `QueryFusionRetriever` re-ranks results from both retrievers using RRF before returning the top-k.

**Best for**: technical documents, datasets with domain-specific terminology.

### 3. HyDE (Hypothetical Document Embeddings)

HyDE is **not a retrieval mode** — it is a query transformation step that happens **before** retrieval in the agent graph (`hyde` node in `node.py`).

Instead of embedding the user's raw question, the LLM first generates a **hypothetical answer**, and that synthetic document is embedded and used as the search query. This dramatically improves recall for complex questions.

```
User question → LLM → Hypothetical answer → embed → Qdrant search → real docs
```

---

## `VectorDBManager`

**`vector_db.py`** — Core class. Manages the full lifecycle of the vector index.

### Storage Layout

```
storage/
  {provider}/                   ← e.g. "ollama/" or "google/"
    qdrant_storage/             ← Qdrant segment files (binary)
    collection_metadata.json    ← Document summaries (JSON)
    {collection_name}/
      docstore.json             ← LlamaIndex node metadata
      index_store.json          ← LlamaIndex index metadata
```

### Key Methods

| Method | Description |
|---|---|
| `add_documents(nodes, collection_name)` | Insert nodes into the index. **Deduplicates** by `node_id` before inserting. Persists metadata after each call. |
| `get_index(collection_name)` | Load an existing `VectorStoreIndex` from disk, or `None` if absent. Handles legacy collection name fallback. |
| `get_retriever(...)` | Returns an `AutoMergingRetriever` for hierarchical search. |
| `get_hybrid_retriever(...)` | Returns a `QueryFusionRetriever` (BM25 + vector + RRF). Falls back to vector-only if docstore is empty. |
| `reset_db()` | **Wipe all data** for the current provider. Required when switching embedding models (dimension mismatch). |
| `save_summary(collection_name, summary)` | Persist a document summary to `collection_metadata.json`. |
| `get_summary(collection_name)` | Read back the summary for a collection. |

### Lazy Initialization

The Qdrant client is **lazy-loaded** via the `db_client` property — it is not created until the first operation that needs it.

```python
@property
def db_client(self) -> QdrantClient:
    if self._db_client is None:
        self._db_client = QdrantClient(path=str(self._qdrant_path))
    return self._db_client
```

### Embedding Dimension Detection

Before creating a Qdrant collection, the manager auto-detects the vector dimension:
1. Checks `embedding_model.embed_dim` attribute (fast path).
2. Falls back to calling `get_text_embedding("dimension test")` and measuring the output length.
3. Defaults to `768` if detection fails (safe for `nomic-embed-text`, `bge-base`).

---

## `Retriever` (High-Level Facade)

**`engine.py`** — Thin wrapper that combines `EmbeddingFactory` + `VectorDBManager` into a single callable interface for the agent layer.

```python
class Retriever:
    def retrieval(
        self,
        hyde: str,                         # The HyDE query from the agent
        collection_name: str,
        k: int,                            # Top-k results
        retrieval_mode: str,               # "hierarchical" | "hybrid"
    ) -> list[NodeWithScore]:
        ...
```

### Usage (from agent tool)

```python
from src.retrieval.engine import Retriever

retriever = Retriever(provider="gemini")
docs = retriever.retrieval(
    hyde="hypothetical answer about climate change...",
    collection_name="my_docs_collection",
    k=5,
    retrieval_mode="hierarchical",
)
```

---

## Configuration

Retrieval parameters are controlled via `src/core/config.py` (`settings.retrieval`):

| Setting | Default | Description |
|---|---|---|
| `retrieval.top_k` | `5` | Number of documents to return |
| `embedding_provider` | `"ollama"` | Drives the `Retriever` provider and storage path |

The `retrieval_mode` is set per-session in `AgentState` and can be overridden at runtime via the API.
