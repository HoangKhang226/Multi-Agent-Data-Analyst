"""LangGraph StateGraph assembly for the Hierarchical RAG pipeline.

Topology
--------
START
  └► retrieve_memory (Mem0)
       └► context_compressor (Summary)
            └► ambiguity_checker
                 ├── is_ambiguous=True  → rejection_handler → update_memory → END
                 └── is_ambiguous=False → planner
                                              └► fan_out_subtasks [Send × N]
                                                    └► subtask_runner (subgraph)
                                                          └► knowledge_router
                                                               ├── "rag"           → hyde → rag_retriever → END
                                                               ├── "web"           → web_searcher → END
                                                               ├── "data_analyzer" → data_analyzer → END
                                                               ├── "visualizer"    → chart_generator → END
                                                               └── "llm_knowledge" → llm_node → END
                                                    (fan-in all branches)
                                              └► synthesizer
                                                   └► update_memory (Mem0 Update)
                                                        └► END
"""

import functools
import inspect
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState, create_initial_state
from src.agents.node import (
    context_compressor,
    ambiguity_checker,
    rejection_handler,
    planner,
    knowledge_router,
    hyde,
    synthesizer,
    llm_node,
)
from src.agents.memory_nodes import retrieve_memory_node, update_memory_node
from src.agents.tool import (
    route_after_ambiguity,
    route_after_router,
    fan_out_subtasks,
    rag_retriever,
    web_searcher,
    data_analyzer,
    chart_generator,
)
from src.utils.logger import logger
import time
from src.db.database import db_manager
from src.db.crud import log_agent_run


# ---------------------------------------------------------------------------
# Log wrapper
# ---------------------------------------------------------------------------

def agent_run_logger(node_func, node_name: str):
    """Wrap a node function to log its execution into the SQL DB.

    Handles both sync and async node callables, including
    ``functools.partial``-wrapped async functions.
    Uses ``db_manager.session()`` for automatic commit/rollback.
    """
    # Unwrap partial so iscoroutinefunction sees the real callable
    _underlying = getattr(node_func, "func", node_func)
    _is_async = inspect.iscoroutinefunction(_underlying)

    @functools.wraps(node_func)
    async def wrapper(state: AgentState):
        t0 = time.time()

        if _is_async:
            result = await node_func(state)
        else:
            result = node_func(state)

        latency_ms = (time.time() - t0) * 1000

        session_id = state.get("session_id")
        if session_id:
            output_preview = str(result)[:500]
            input_preview = str(state.get("current_task") or state.get("question") or "")[:500]

            status = "ok"
            if isinstance(result, dict):
                for res in result.get("sub_task_results", []):
                    if isinstance(res, dict) and res.get("type") == "error":
                        status = "error"
                        break

            try:
                with db_manager.session() as db_session:
                    log_agent_run(
                        db_session, session_id, node_name,
                        input_preview, output_preview, latency_ms, status,
                    )
            except Exception as e:
                logger.error(f"Failed to log agent run for {node_name}: {e}")

        return result
    return wrapper

# ---------------------------------------------------------------------------
# Sub-task subgraph
# ---------------------------------------------------------------------------

def build_subtask_subgraph(df=None):
    """Compile the per-sub-task retrieval mini-pipeline."""
    sg = StateGraph(AgentState)

    # Bind df to specialized nodes
    bound_analyzer = functools.partial(data_analyzer, df=df)
    bound_visualizer = functools.partial(chart_generator, df=df)

    # add node but wrap log
    sg.add_node("knowledge_router", agent_run_logger(knowledge_router, "knowledge_router"))
    sg.add_node("hyde", agent_run_logger(hyde, "hyde"))
    sg.add_node("rag_retriever", agent_run_logger(rag_retriever, "rag_retriever"))
    sg.add_node("web_searcher", agent_run_logger(web_searcher, "web_searcher"))
    sg.add_node("llm_node", agent_run_logger(llm_node, "llm_node"))
    sg.add_node("data_analyzer", agent_run_logger(bound_analyzer, "data_analyzer"))
    sg.add_node("chart_generator", agent_run_logger(bound_visualizer, "chart_generator"))

    sg.set_entry_point("knowledge_router")

    # Routing from knowledge_router
    sg.add_conditional_edges(
        "knowledge_router",
        route_after_router,
        {
            "hyde": "hyde",
            "web_searcher": "web_searcher",
            "llm_node": "llm_node",
            "data_analyzer": "data_analyzer",
            "chart_generator": "chart_generator",
        },
    )

    sg.add_edge("hyde", "rag_retriever")
    sg.add_edge("rag_retriever", END)

    sg.add_edge("web_searcher", END)
    sg.add_edge("llm_node", END)
    sg.add_edge("data_analyzer", END)
    sg.add_edge("chart_generator", END)

    return sg.compile()


# ---------------------------------------------------------------------------
# Main graph factory
# ---------------------------------------------------------------------------


def build_graph(df=None):
    """Compile and return the top-level Hierarchical RAG StateGraph."""
    subtask_subgraph = build_subtask_subgraph(df=df)

    g = StateGraph(AgentState)

    # ---- Nodes ----
    g.add_node("retrieve_memory", retrieve_memory_node)
    g.add_node("context_compressor", context_compressor)
    g.add_node("ambiguity_checker", ambiguity_checker)
    g.add_node("rejection_handler", rejection_handler)
    g.add_node("planner", planner)
    g.add_node("subtask_runner", subtask_subgraph)
    g.add_node("synthesizer", synthesizer)
    g.add_node("update_memory", update_memory_node)

    # ---- Edges ----
    g.set_entry_point("retrieve_memory")
    g.add_edge("retrieve_memory", "context_compressor")
    g.add_edge("context_compressor", "ambiguity_checker")

    # Ambiguity gate
    g.add_conditional_edges(
        "ambiguity_checker",
        route_after_ambiguity,
        {
            "planner": "planner",
            "rejection_handler": "rejection_handler",
        },
    )

    g.add_edge("rejection_handler", "update_memory")

    # Fan-out
    g.add_conditional_edges("planner", fan_out_subtasks, ["subtask_runner"])

    # Fan-in
    g.add_edge("subtask_runner", "synthesizer")
    g.add_edge("synthesizer", "update_memory")
    g.add_edge("update_memory", END)

    compiled = g.compile()
    logger.info("[build_graph] LangGraph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Initial state helper (exported for use by business logic)
# ---------------------------------------------------------------------------


def make_initial_state(
    provider: str = None,
    memory_provider: str = None,
    collection_name: str = None,
    user_id: str = None,
    data_mode: str = None,
    retrieval_mode: str = "hierarchical",
) -> AgentState:
    """Return a fresh AgentState with overridden defaults."""
    state = create_initial_state()
    
    if provider:
        state["llm_provider"] = provider
    if memory_provider:
        state["memory_provider"] = memory_provider
    if collection_name:
        state["collection_name"] = collection_name
    if user_id:
        state["user_id"] = user_id
    if data_mode:
        state["data_mode"] = data_mode
    if retrieval_mode:
        state["retrieval_mode"] = retrieval_mode
        
    return state


# Default compiled graph
graph = build_graph()
