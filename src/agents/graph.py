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
from typing import Optional

from langgraph.graph import StateGraph, END, START

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


# ---------------------------------------------------------------------------
# Sub-task subgraph
# ---------------------------------------------------------------------------


def build_subtask_subgraph(df=None):
    """Compile the per-sub-task retrieval mini-pipeline."""
    sg = StateGraph(AgentState)

    # Bind df to specialized nodes
    bound_analyzer = functools.partial(data_analyzer, df=df)
    bound_visualizer = functools.partial(chart_generator, df=df)

    sg.add_node("knowledge_router", knowledge_router)
    sg.add_node("hyde", hyde)
    sg.add_node("rag_retriever", rag_retriever)
    sg.add_node("web_searcher", web_searcher)
    sg.add_node("llm_node", llm_node)
    sg.add_node("data_analyzer", bound_analyzer)
    sg.add_node("chart_generator", bound_visualizer)

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
