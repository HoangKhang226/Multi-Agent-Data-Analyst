"""Tool functions and conditional-edge routing for the Hierarchical RAG pipeline.

Production Refactor:
- Tool nodes return raw data (TaskResult) instead of textual explanations.
- Reasoning is deferred to the synthesizer.
"""

from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import uuid
import re

from langgraph.graph import END
from langgraph.types import Send

from langchain_community.tools.tavily_search import TavilySearchResults

from src.agents.state import AgentState, TaskResult
from src.llm.factory import LLMFactory
from src.llm.embeddings import EmbeddingFactory
from src.retrieval.vector_db import VectorDBManager
from src.retrieval.engine import Retriever
from src.core.config import settings
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _provider(state: AgentState) -> Optional[str]:
    return state.get("llm_provider") or settings.graph_provider


def _get_llm(provider: Optional[str], purpose: str):
    return LLMFactory.create_client(purpose=purpose, provider=provider).get_llm()


def _get_vector_db(provider: Optional[str]) -> VectorDBManager:
    embedding = EmbeddingFactory().get_embedding(provider=provider)
    return VectorDBManager(embedding_model=embedding, provider=provider)


# ---------------------------------------------------------------------------
# Routing — conditional edge functions
# ---------------------------------------------------------------------------


def route_after_ambiguity(state: AgentState) -> str:
    if state.get("is_ambiguous"):
        return "rejection_handler"
    return "planner"


def route_after_router(state: AgentState) -> str:
    route = state.get("route", "llm_knowledge").strip().lower()
    if route == "rag":
        has_doc = bool((state.get("content_summary") or "").strip())
        if not has_doc:
            return "data_analyzer"

    mapping = {
        "rag": "hyde",
        "web": "web_searcher",
        "llm_knowledge": "llm_node",
        "data_analyzer": "data_analyzer",
        "visualizer": "chart_generator",
    }
    return mapping.get(route, "llm_knowledge")


# ---------------------------------------------------------------------------
# Fan-out — REFACTORED for minimal state propagation
# ---------------------------------------------------------------------------


def fan_out_subtasks(state: AgentState) -> List[Send]:
    """Dispatch parallel sub-tasks with isolated state."""
    sub_tasks = state.get("sub_tasks") or []
    
    # We only pass required fields to the branched state
    # Note: we include llm_provider/collection_name so sub-task tools know where to run
    base_payload = {
        "llm_provider": state.get("llm_provider"),
        "collection_name": state.get("collection_name"),
        "dataframe_head": state.get("dataframe_head"),
        "user_id": state.get("user_id"),
        "content_summary": state.get("content_summary"),
        "user_memory": state.get("user_memory"),
        "data_mode": state.get("data_mode"),
        "retrieval_mode": state.get("retrieval_mode"),
    }

    return [
        Send("subtask_runner", {**base_payload, "current_task": task})
        for task in sub_tasks
    ]


# ---------------------------------------------------------------------------
# Node 5a — RAG Retriever (REFACTORED: Returns RAW chunks)
# ---------------------------------------------------------------------------


def rag_retriever(state: AgentState) -> dict:
    """Retrieve raw chunks; no interpretation."""
    provider = _provider(state)
    current_task = state.get("current_task")
    if not current_task:
        return {}

    query = state.get("hyde_query") or current_task["description"]
    collection_name = state.get("collection_name") or settings.storage.collection_name

    try:
        retriever_obj = Retriever(provider=provider)
        nodes = retriever_obj.retrieval(
            hyde=query,
            collection_name=collection_name,
            k=settings.retrieval.top_k,
            retrieval_mode=state.get("retrieval_mode", "hierarchical")
        )
        
        chunks = [node.get_content() for node in nodes if node.get_content().strip()]
        
        content = "\n\n---\n\n".join(chunks)
        
        result_obj: TaskResult = {
            "task": current_task["description"],
            "type": "text", 
            "content": f"[DATA FROM DOCUMENTS]:\n{content}"
        }

        return {
            "all_context": chunks,
            "sub_task_results": [result_obj]
        }

    except Exception as e:
        logger.error(f"[rag_retriever] Error: {e}")
        return {
            "all_context": [],
            "sub_task_results": [{"type": "error", "content": f"RAG retrieval failed: {e}"}]
        }


# ---------------------------------------------------------------------------
# Node 5b — Web Searcher (REFACTORED: Returns RAW snippets)
# ---------------------------------------------------------------------------


def web_searcher(state: AgentState) -> dict:
    """Search web; returns raw results list."""
    current_task = state.get("current_task")
    if not current_task:
        return {}

    query = current_task["description"]

    try:
        tool = TavilySearchResults(max_results=5)
        results = tool.invoke(query)
        
        result_obj: TaskResult = {
            "task": query,
            "type": "text",
            "content": f"WEB_SEARCH_RESULTS for '{query}':\n\n{results}"
        }
        return {"sub_task_results": [result_obj]}

    except Exception as e:
        logger.error(f"[web_searcher] Error: {e}")
        return {"sub_task_results": [{"type": "error", "content": f"Web search failed: {e}"}]}


# ---------------------------------------------------------------------------
# Node 5c — LLM Knowledge Node (REFACTORED: Returns raw LLM text)
# ---------------------------------------------------------------------------


def llm_node_stub(state: AgentState) -> dict:
    """Old llm_node logic removed from tool.py. Original is in node.py."""
    pass


# ---------------------------------------------------------------------------
# Node 5d — Data Analyzer (Numerical statistics)
# ---------------------------------------------------------------------------


def data_analyzer(state: AgentState, df: Optional[pd.DataFrame] = None) -> dict:
    """LLM Generates Numerical Code -> exec() -> Return JSON result."""
    from src.prompt.template import DATA_ANALYZER_PROMPT
    
    current_task = state.get("current_task")
    if not current_task:
        return {}

    task_desc = current_task["description"]
    dataframe_head = state.get("dataframe_head") or ""
    provider = _provider(state)

    if df is None:
        return {"sub_task_results": [{"type": "error", "content": "No DataFrame available."}]}

    try:
        llm = _get_llm(provider, "rag")
        prompt = DATA_ANALYZER_PROMPT.format(
            task=task_desc,
            dataframe_head=dataframe_head or df.head(5).to_string(),
            dataframe_info=state.get("dataframe_info") or "N/A",
        )
        response = llm.invoke(prompt)
        raw_code = response.content if hasattr(response, "content") else str(response)

        code_match = re.search(r"```(?:python)?\n(.*?)```", raw_code, re.DOTALL)
        code = code_match.group(1).strip() if code_match else raw_code.strip()

        exec_globals = {
            "df": df.copy(),
            "pd": pd,
            "__builtins__": __builtins__,
        }

        exec(code, exec_globals)  # noqa: S102

        result = exec_globals.get("result", None)

        if result is None:
            return {
                "sub_task_results": [{
                    "type": "error",
                    "content": "No 'result' variable returned from data_analyzer code."
                }]
            }

        return {"sub_task_results": [{"task": task_desc, "type": "json", "content": result}]}

    except Exception as e:
        logger.error(f"[data_analyzer] Error: {e}")
        error_msg = str(e)
        if isinstance(e, KeyError):
            error_msg = f"Column {e} does not exist. Available columns: {list(df.columns)}"
        
        return {
            "sub_task_results": [{
                "type": "error",
                "content": f"Data analysis failed: {error_msg}"
            }]
        }


# ---------------------------------------------------------------------------
# Node 5e — Visualizer (Charts + Stats)
# ---------------------------------------------------------------------------


def chart_generator(state: AgentState, df: Optional[pd.DataFrame] = None) -> dict:
    """LLM Generates Visualization Code -> exec() -> Return Chart + Stats."""
    from src.prompt.template import VISUALIZER_PROMPT
    
    current_task = state.get("current_task")
    if not current_task:
        return {}

    task_desc = current_task["description"]
    dataframe_head = state.get("dataframe_head") or ""
    provider = _provider(state)

    if df is None:
        return {"sub_task_results": [{"type": "error", "content": "No DataFrame available."}]}

    try:
        charts_dir = Path("output_charts")
        charts_dir.mkdir(exist_ok=True)
        chart_path = charts_dir / f"chart_{uuid.uuid4().hex[:8]}.png"

        llm = _get_llm(provider, "rag")
        prompt = VISUALIZER_PROMPT.format(
            task=task_desc,
            dataframe_head=dataframe_head or df.head(5).to_string(),
            dataframe_info=state.get("dataframe_info") or "N/A",
            chart_path=chart_path.as_posix(),
        )
        response = llm.invoke(prompt)
        raw_code = response.content if hasattr(response, "content") else str(response)

        code_match = re.search(r"```(?:python)?\n(.*?)```", raw_code, re.DOTALL)
        code = code_match.group(1).strip() if code_match else raw_code.strip()

        exec_globals = {
            "df": df.copy(),
            "pd": pd,
            "__builtins__": __builtins__,
        }

        exec(code, exec_globals)  # noqa: S102

        result = exec_globals.get("result", None)
        saved_chart_path = str(chart_path) if chart_path.exists() else None

        results: List[TaskResult] = []
        if result:
            results.append({"task": task_desc, "type": "json", "content": result})
        
        if saved_chart_path:
            results.append({"task": task_desc, "type": "chart", "content": saved_chart_path})

        if not results:
            return {
                "sub_task_results": [{
                    "type": "error",
                    "content": "No 'result' or chart produced by visualizer."
                }]
            }

        return {"sub_task_results": results}

    except Exception as e:
        logger.error(f"[chart_generator] Error: {e}")
        error_msg = str(e)
        if "None of" in error_msg and "are in the [columns]" in error_msg:
            error_msg = f"Requested column not found in data. Available columns: {list(df.columns)}"
        elif isinstance(e, KeyError):
            error_msg = f"Column {e} does not exist. Available columns: {list(df.columns)}"

        return {
            "sub_task_results": [{
                "type": "error",
                "content": f"Visualization failed: {error_msg}"
            }]
        }
