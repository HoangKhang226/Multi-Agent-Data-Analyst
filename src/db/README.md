# `src/db/` — SQL Persistence Layer

This module provides structured, relational persistence for the Chat With Data application using **SQLAlchemy** (ORM) with **SQLite** as the default backend (easily swappable for PostgreSQL).

---

## File Overview

| File | Responsibility |
|---|---|
| `database.py` | `DatabaseManager` class — engine, session factory, schema management, FastAPI dependency |
| `models.py` | SQLAlchemy ORM models (table definitions + relationships) |
| `crud.py` | Data access functions (create/read helpers for each entity) |
| `__init__.py` | Package bootstrapper — imports models so they register with `Base.metadata` before `init_db()` |

---

## Schema

```
users
  │  user_id (PK, unique string)
  │  created_at
  │
  ├──► datasets (1:N)
  │      │  user_id (FK → users.user_id)
  │      │  filename
  │      │  collection_name     ← Qdrant collection name
  │      │  provider            ← embedding provider used
  │      │  data_mode           ← "document" | "tabular"
  │      │  chunks_count
  │      │  summary
  │      │  created_at
  │      │
  │      └──► chat_sessions (1:N via dataset_id)
  │
  └──► chat_sessions (1:N)
         │  user_id (FK → users.user_id)
         │  dataset_id (FK → datasets.id, NULLABLE)
         │  created_at
         │
         ├──► messages (1:N)
         │      session_id (FK → chat_sessions.id)
         │      role          ← "user" | "assistant"
         │      content
         │      tokens_used
         │      created_at
         │
         └──► agent_runs (1:N)
                session_id (FK → chat_sessions.id)
                node_name     ← LangGraph node identifier
                input_preview ← first 500 chars of node input
                output_preview← first 500 chars of node output
                latency_ms    ← wall-clock execution time
                status        ← "ok" | "error"
                created_at
```

### Entity Relationships

```
User ──< Dataset ──< ChatSession ──< Message
                          └────────< AgentRun
```

- A `User` can have multiple `Dataset`s and multiple `ChatSession`s.
- A `ChatSession` is optionally linked to a `Dataset` (a pure chat session has `dataset_id = NULL`).
- Each `ChatSession` accumulates `Message` rows (the conversation history) and `AgentRun` rows (the execution trace of every LangGraph node).

---

## `DatabaseManager` (OOP Class)

`database.py` wraps all SQLAlchemy boilerplate into a single class:

```python
db_manager = DatabaseManager(url="sqlite:///storage/app.db")
db_manager.init_db()   # CREATE TABLE IF NOT EXISTS …

# Context-manager session (auto commit/rollback)
with db_manager.session() as db:
    db.add(User(user_id="alice"))

# FastAPI dependency
@app.get("/")
def route(db: Session = Depends(db_manager.get_db)):
    ...
```

### Session Modes

| Method | Use case | Lifecycle |
|---|---|---|
| `session()` | Background tasks, agent logger | `with` block — auto-commit on exit, auto-rollback on exception |
| `get_db()` | FastAPI route dependency | Generator — session per HTTP request, closed in `finally` |
| `get_session()` | Manual control | Caller must `.close()` explicitly |

### Health Check

```python
db_manager.health_check()  # → True / False
```

Executes `SELECT 1` against the engine. Used by the `/health` API endpoint.

---

## CRUD Functions (`crud.py`)

| Function | Description |
|---|---|
| `get_or_create_user(db, user_id)` | Upsert a `User` row by `user_id` string. |
| `create_session(db, user_id, dataset_id)` | Open a new `ChatSession`. Returns the session `id`. |
| `log_message(db, session_id, role, content)` | Append a `Message` row to a session. |
| `log_agent_run(db, session_id, node_name, ...)` | Write an `AgentRun` trace row. Called automatically by `agent_run_logger` in `graph.py`. |

---

## Startup Sequence

The database is initialized once at **application startup** in `src/api/main.py`:

```python
from src.db import init_db
init_db()   # Creates all tables if they don't exist
```

The `src/db/__init__.py` imports all model modules **before** `init_db()` is called, so SQLAlchemy's `Base.metadata` has all table definitions registered:

```python
# __init__.py
from src.db.models import User, Dataset, ChatSession, Message, AgentRun  # noqa
from src.db.database import init_db, get_db, db_manager
```

---

## Configuration

The database URL is read from `src/core/config.py`:

```python
settings.database.url   # default: "sqlite:///storage/app.db"
```

To switch to PostgreSQL, set the env variable:
```
DATABASE_URL=postgresql://user:pass@host/dbname
```

The `DatabaseManager` automatically adjusts `connect_args` — `check_same_thread=False` is only applied for SQLite.

---

## Agent Run Logging

Every LangGraph node execution is automatically logged to `agent_runs` via the `agent_run_logger` decorator in `graph.py`:

```python
# In graph.py — wraps every node at compile time
sg.add_node("knowledge_router", agent_run_logger(knowledge_router, "knowledge_router"))
```

This gives full **execution traceability** per session: which nodes ran, how long they took, and whether they errored.
