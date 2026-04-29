import operator
from typing import Annotated, List, Literal, Optional, TypedDict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ==============================
# Reducers
# ==============================

def write_once(old, new):
    """Keep the first **non-None** value assigned to the field.

    Treating ``None`` as "not yet written" lets the API layer inject
    identifiers such as ``session_id`` into a freshly-created state dict
    without being blocked by the initial ``None`` default.
    """
    if old is not None:  # Already has a real value — keep it
        return old
    return new  # old is None → accept the incoming value

def safe_add_list(old, new):
    """Parallel-safe list concatenation."""
    if new is None: # If new is None, return old list or empty
        return old or []
    if not isinstance(new, list): # If new is not a list, wrap it
        new = [new]
    return (old or []) + new

# ==============================
# Sub-task & Result schemas
# ==============================

class SubTask(TypedDict):
    task_type: Literal["data_analyzer", "visualizer", "rag", "web_search", "llm_knowledge"]
    description: str
    status: Literal["pending", "completed", "failed"]
    result: Optional[str]

class TaskResult(TypedDict):
    """Production-grade structured result from a sub-task."""
    task: str
    type: Literal["stats", "chart", "table", "text", "json", "error"]
    content: Any  # Raw data (str, dict, list, path)

# ==============================
# Agent State (Production Ready)
# ==============================

class AgentState(TypedDict):

    # ------------------------------------------------------------------
    # 1. Session & Identity (READ ONLY after init)
    # ------------------------------------------------------------------

    session_id: Annotated[Optional[int], write_once]
    user_id: Annotated[Optional[str], write_once]
    llm_provider: Annotated[Optional[str], write_once]
    memory_provider: Annotated[Optional[str], write_once]
    collection_name: Annotated[Optional[str], write_once]
    
    # NEW: Data mode (document | tabular | None)
    data_mode: Annotated[Optional[Literal["document", "tabular"]], write_once]
    
    # NEW: Retrieval strategy for documents
    retrieval_mode: Annotated[Optional[Literal["hierarchical", "hybrid"]], write_once]

    # ------------------------------------------------------------------
    # 2. Input & Context (READ ONLY after init)
    # ------------------------------------------------------------------

    question: Annotated[Optional[str], write_once]
    input_data: Annotated[Optional[str], write_once]
    dataframe_head: Annotated[Optional[str], write_once]
    dataframe_info: Annotated[Optional[str], write_once]
    content_summary: Annotated[Optional[str], write_once]
    user_memory: Annotated[Optional[str], write_once]

    # ------------------------------------------------------------------
    # 3. Accumulators (PARALLEL SAFE - APPEND ONLY)
    # ------------------------------------------------------------------

    messages: Annotated[List[BaseMessage], add_messages]

    tool_outputs: Annotated[List[str], safe_add_list]
    intermediate_steps: Annotated[List[str], safe_add_list]
    error_logs: Annotated[List[str], safe_add_list]

    all_context: Annotated[List[str], safe_add_list]
    
    # REFACTORED: From List[str] to List[TaskResult]
    sub_task_results: Annotated[List[TaskResult], safe_add_list]
    
    chart_paths: Annotated[List[str], safe_add_list]

    # ------------------------------------------------------------------
    # 4. Control Flow (SINGLE WRITER)
    # ------------------------------------------------------------------

    is_ambiguous: Annotated[Optional[bool], write_once]
    rejection_reason: Annotated[Optional[str], write_once]

    sub_tasks: Annotated[Optional[List[SubTask]], write_once]

    route: Annotated[Optional[Literal[
        "rag", "web", "llm_knowledge", "data_analyzer", "visualizer", "ambiguous", "direct_answer"
    ]], write_once]

    retry_count: Annotated[int, operator.add]

    # ------------------------------------------------------------------
    # 5. Per-task Execution (ISOLATED PER SEND)
    # ------------------------------------------------------------------

    current_task: Annotated[Optional[SubTask], write_once]

    hyde_query: Annotated[Optional[str], write_once]
    
    is_context_valid: Annotated[Optional[bool], write_once]
    validation_score: Annotated[Optional[float], write_once]

    # ------------------------------------------------------------------
    # 6. Final Output (WRITE ONCE)
    # ------------------------------------------------------------------

    final_answer: Annotated[Optional[str], write_once]


# ==============================
# Initial State Helper
# ==============================

def create_initial_state() -> AgentState:
    return {
        "session_id": None,
        "user_id": "guest",
        "llm_provider": None,
        "memory_provider": None,
        "collection_name": None,
        "data_mode": None,
        "retrieval_mode": "hierarchical",

        "question": None,
        "input_data": None,
        "dataframe_head": None,
        "dataframe_info": None,
        "content_summary": None,
        "user_memory": None,

        "messages": [],

        "tool_outputs": [],
        "intermediate_steps": [],
        "error_logs": [],

        "all_context": [],
        "sub_task_results": [],
        "chart_paths": [],

        "is_ambiguous": None,
        "rejection_reason": None,

        "sub_tasks": None,
        "route": None,

        "retry_count": 0,

        "current_task": None,
        "hyde_query": None,

        "is_context_valid": None,
        "validation_score": None,

        "final_answer": None,
    }