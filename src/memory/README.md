# `src/memory/` — Long-Term User Memory

This module implements **persistent, cross-session user memory** using [Mem0](https://mem0.ai/). It extracts structured facts from conversations, stores them in a local ChromaDB vector store, and retrieves relevant facts at the start of each new session — giving the agent a "memory" of who the user is and what they care about.

---

## File Overview

| File | Responsibility |
|---|---|
| `long_term.py` | `LongTermMemoryManager` class — fact extraction, storage, and retrieval via Mem0 |

---

## Architecture

```
Conversation turn (question + answer)
          │
          ▼
  LongTermMemoryManager.extract_facts()
          │  Structured LLM call → FactList (Pydantic)
          │  Filters: content length > 5 chars
          ▼
  Mem0.add(fact, user_id, metadata)
          │
          └─► ChromaDB (local vector store)
                 collection: "mem0_user_memory"
                 path: settings.memory.chroma_path

─────────── Next session ───────────

  LongTermMemoryManager.get_user_facts(query, user_id)
          │
          └─► Mem0.search(query, filters={"user_id": ...}, limit=5)
                 └─► Returns formatted bullet list → injected into AgentState.user_memory
```

---

## Fact Schema

Facts are extracted and stored as typed, structured objects:

```python
class FactMetadata(BaseModel):
    type: Literal["preference", "habit", "insight", "skill", "goal", "identity"]
    confidence: float  # 0.0 – 1.0

class Fact(BaseModel):
    content: str        # The fact text, min 5 chars
    metadata: FactMetadata
```

**Fact types:**

| Type | Example |
|---|---|
| `preference` | "User prefers concise answers without bullet points" |
| `habit` | "User typically uploads CSV files for financial analysis" |
| `insight` | "User is interested in revenue trends for Q3 2024" |
| `skill` | "User is familiar with pandas and matplotlib" |
| `goal` | "User wants to automate monthly reporting" |
| `identity` | "User works as a data analyst at a retail company" |

---

## `LongTermMemoryManager`

### Multi-provider Support

The manager supports two LLM/embedding backends, configured per-call or via `settings.memory_provider`:

| Provider | LLM model | Embedding model |
|---|---|---|
| `"ollama"` | `settings.ollama.model` | `settings.ollama.embed_model` |
| `"gemini"` | `settings.gemini.model` | `settings.embedding.google` |

Mem0 instances are **cached per provider** in `_mem0_instances` — each provider is initialized only once (lazy loading):

```python
mem0_manager.get_mem0(provider="gemini")   # created on first call, reused after
```

### Key Methods

| Method | Description |
|---|---|
| `get_user_facts(query, user_id, provider)` | Search Mem0 for facts relevant to `query` for the given user. Returns a formatted bullet string (max 5 facts). |
| `save_user_facts(messages, user_id, provider)` | Extract facts from the last Q&A pair in `messages` and persist them to Mem0. |
| `extract_facts(question, answer, provider)` | Internal — runs a structured LLM prompt to extract `Fact` objects from a Q&A pair. |
| `get_mem0(provider)` | Return the `Memory` instance for the given provider, initializing it if necessary. |

### Async Design

All public methods are `async`. The underlying Mem0 `search` and `add` calls are wrapped in `asyncio.to_thread()` to avoid blocking the event loop (Mem0's SDK is synchronous):

```python
results = await asyncio.to_thread(m.search, query=query, filters={"user_id": user_id})
```

---

## Integration with the Agent Graph

Memory is injected into the agent pipeline via two dedicated nodes in `graph.py`:

### `retrieve_memory` (start of pipeline)
```python
# memory_nodes.py
async def retrieve_memory_node(state: AgentState) -> dict:
    facts = await memory_manager.get_user_facts(
        query=state["question"],
        user_id=state["user_id"],
        provider=state.get("memory_provider"),
    )
    return {"user_memory": facts}
```
The `user_memory` string is then referenced by `ambiguity_checker`, `planner`, `knowledge_router`, `llm_node`, and `synthesizer` via `_build_memory_section()`.

### `update_memory` (end of pipeline)
```python
# memory_nodes.py
async def update_memory_node(state: AgentState) -> dict:
    messages = [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state["final_answer"]},
    ]
    await memory_manager.save_user_facts(messages, state["user_id"])
    return {}
```

---

## Storage Layout

```
storage/
  memory/
    chroma/           ← ChromaDB vector store (Mem0 embeddings)
      mem0_user_memory/
    mem0_history.db   ← Mem0 history SQLite (operation log)
```

Paths are configured in `settings.memory`:
- `settings.memory.chroma_path`
- `settings.memory.history_db_path`

---

## Singleton

A module-level singleton is exported for use across the codebase:

```python
# long_term.py
memory_manager = LongTermMemoryManager()

# Convenience aliases
get_user_facts = memory_manager.get_user_facts
save_user_facts = memory_manager.save_user_facts
```

Import from anywhere:
```python
from src.memory.long_term import get_user_facts, save_user_facts
```
