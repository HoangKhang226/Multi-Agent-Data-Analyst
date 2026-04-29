# `src/agents/` — LangGraph Orchestration Layer

This module contains the complete **Hierarchical Multi-Agent RAG pipeline** built with [LangGraph](https://langchain-ai.github.io/langgraph/). It defines the graph topology, all agent nodes, routing logic, and the shared state schema.

---

## File Overview

| File | Responsibility |
|---|---|
| `graph.py` | Assembles and compiles the full `StateGraph`; defines the sub-task subgraph |
| `node.py` | Implements all LangGraph node functions (the "brains" of each step) |
| `tool.py` | Implements tool-nodes (RAG retriever, web search, data analyzer, chart generator) and routing functions |
| `memory_nodes.py` | Handles Mem0 memory retrieval (`retrieve_memory`) and persistence (`update_memory`) |
| `state.py` | Defines `AgentState` TypedDict, reducers, and `SubTask`/`TaskResult` schemas |

---

## Graph Topology

The pipeline is a two-level nested graph: a **main graph** that orchestrates the full conversation turn, and a **subgraph** that runs each decomposed sub-task in parallel.

```
START
  └─► retrieve_memory          ← Fetch relevant user facts from Mem0
        └─► context_compressor ← Summarize uploaded document (once per session)
              └─► ambiguity_checker
                    │
                    ├─ [is_ambiguous=True]  → rejection_handler → update_memory → END
                    │
                    └─ [is_ambiguous=False] → planner
                                                  └─► fan_out_subtasks (Send × N, parallel)
                                                        └─► subtask_runner (subgraph per task)
                                                              └─► knowledge_router
                                                                    ├─ "rag"           → hyde → rag_retriever → END
                                                                    ├─ "web"           → web_searcher        → END
                                                                    ├─ "data_analyzer" → data_analyzer       → END
                                                                    ├─ "visualizer"    → chart_generator     → END
                                                                    └─ "llm_knowledge" → llm_node            → END
                                                        (fan-in: all branches merge back)
                                                  └─► synthesizer      ← Heavy reasoning + formatting
                                                        └─► update_memory → END
```

---

## Node Descriptions

### Main Graph Nodes

| Node | File | What it does |
|---|---|---|
| `retrieve_memory` | `memory_nodes.py` | Calls Mem0 to fetch up to 5 relevant user facts for the current query. Injects them into `user_memory` in state. |
| `context_compressor` | `node.py` | Summarizes the uploaded document into `content_summary`. Runs **once per session** (skipped if summary exists). |
| `ambiguity_checker` | `node.py` | Uses a structured LLM call to decide if the question is too vague to answer. Sets `is_ambiguous` and `rejection_reason`. |
| `rejection_handler` | `node.py` | Writes a polite clarification request to `final_answer`. Triggered only when `is_ambiguous=True`. |
| `planner` | `node.py` | Decomposes the user question into a list of `SubTask` objects (each with a `task_type` and `description`). |
| `subtask_runner` | `graph.py` | A compiled sub-graph; one instance is spawned per sub-task via `Send()`. Runs the retrieval pipeline for that task. |
| `synthesizer` | `node.py` | **The heavy reasoning node.** Receives all `TaskResult` objects from parallel branches, merges them, and writes the final Markdown answer to `final_answer`. |
| `update_memory` | `memory_nodes.py` | Saves the Q&A turn back to Mem0 so future sessions benefit from it. |

### Subgraph Nodes (inside `subtask_runner`)

| Node | File | What it does |
|---|---|---|
| `knowledge_router` | `node.py` | Classifies each sub-task and selects the appropriate retrieval strategy. Routes are filtered by `data_mode`. |
| `hyde` | `node.py` | Generates a **Hypothetical Document Embedding** — a synthetic answer used as a better semantic search query. |
| `rag_retriever` | `tool.py` | Runs the actual vector search against Qdrant using the HyDE query. Returns ranked document chunks. |
| `web_searcher` | `tool.py` | Performs a live web search (DuckDuckGo / Tavily). Used when no document is uploaded. |
| `data_analyzer` | `tool.py` | Runs pandas/statistical analysis on the uploaded DataFrame. Produces `stats` or `table` results. |
| `chart_generator` | `tool.py` | Generates matplotlib/plotly charts from the DataFrame. Saves chart files and returns paths. |
| `llm_node` | `node.py` | Falls back to pure LLM knowledge when no external data source is needed or available. |

---

## Routing Logic

### 1. `route_after_ambiguity` — after `ambiguity_checker`

```python
"planner"           # is_ambiguous == False
"rejection_handler" # is_ambiguous == True
```

### 2. `fan_out_subtasks` — after `planner`

Uses LangGraph's `Send()` primitive to dispatch **each sub-task** to a separate `subtask_runner` instance. This is the fan-out mechanism that enables parallelism.

### 3. `route_after_router` — inside subgraph, after `knowledge_router`

The router first selects a route from `{rag, web, llm_knowledge, data_analyzer, visualizer}`, then **filters by `data_mode`**:

| `data_mode` | Allowed routes | Fallback |
|---|---|---|
| `"document"` | `rag`, `web`, `llm_knowledge` | `rag` |
| `"tabular"` | `data_analyzer`, `visualizer`, `llm_knowledge` | `data_analyzer` |
| `None` | `llm_knowledge` | `llm_knowledge` |

---

## AgentState

Defined in `state.py`. Uses custom **reducers** to control how fields merge during parallel execution:

| Reducer | Behaviour | Used by |
|---|---|---|
| `write_once` | Keeps the first non-`None` value; ignores subsequent writes | `session_id`, `user_id`, `question`, `route`, etc. |
| `safe_add_list` | Thread-safe append — never overwrites, always concatenates | `sub_task_results`, `chart_paths`, `tool_outputs` |
| `add_messages` | LangChain standard message accumulator | `messages` |
| `operator.add` | Integer accumulation | `retry_count` |

### State Field Groups

```
Session & Identity   → session_id, user_id, llm_provider, data_mode, retrieval_mode
Input & Context      → question, input_data, dataframe_head, content_summary, user_memory
Accumulators         → messages, sub_task_results, chart_paths, tool_outputs
Control Flow         → is_ambiguous, sub_tasks, route, retry_count
Per-task Execution   → current_task, hyde_query
Final Output         → final_answer
```

---

## Observability

Every node is wrapped by `agent_run_logger()` in `graph.py`. This decorator:
- Measures **latency** (`ms`) for each node execution
- Detects **error** status from `sub_task_results`
- Writes an `AgentRun` row to the SQL database via `src.db.crud.log_agent_run`
- Is transparent to both sync and async node functions (handles `functools.partial` too)

---

## Usage

```python
from src.agents.graph import build_graph, make_initial_state

graph = build_graph(df=my_dataframe)  # df=None for document/chat-only mode

state = make_initial_state(
    provider="gemini",
    user_id="user_123",
    data_mode="document",
    collection_name="my_docs",
)
state["question"] = "What are the key findings?"
state["session_id"] = 42

result = await graph.ainvoke(state)
print(result["final_answer"])
```
