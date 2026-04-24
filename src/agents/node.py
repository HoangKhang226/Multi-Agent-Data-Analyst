"""LangGraph node implementations for the Hierarchical RAG pipeline.

Production Refactor:
- Nodes return raw data (TaskResult) to the state.
- Centralized reasoning in the Synthesizer node.
"""

import json
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

from langchain_core.messages import AIMessage

from src.agents.state import AgentState, SubTask, TaskResult
from src.core.config import settings
from src.prompt.template import (
    CONTEXT_COMPRESSION_PROMPT,
    AMBIGUITY_CHECK_PROMPT,
    PLANNER_PROMPT,
    KNOWLEDGE_ROUTER_PROMPT,
    HYDE_PROMPT,
    SYNTHESIZER_PROMPT,
    USER_MEMORY_SECTION,
    REJECTION_FALLBACK_ANSWER,
    TECHNICAL_ERROR_RESPONSE,
    LLM_KNOWLEDGE_DOC_PROMPT,
    LLM_KNOWLEDGE_TAB_PROMPT,
    LLM_KNOWLEDGE_BASE_PROMPT,
)
from src.utils.logger import logger
from src.llm.factory import LLMFactory


# ---------------------------------------------------------------------------
# Pydantic Output Schemas
# ---------------------------------------------------------------------------


class AmbiguityCheckOutput(BaseModel):
    is_ambiguous: bool = Field(description="True if question is vague.")
    reason: str = Field(default="", description="Reason for the decision.")


class SubTaskSchema(BaseModel):
    task_type: Literal["data_analyzer", "visualizer", "rag", "web_search", "llm_knowledge"]
    description: str


class PlannerOutput(BaseModel):
    sub_tasks: List[SubTaskSchema]


class KnowledgeRouterOutput(BaseModel):
    route: Literal["rag", "web", "llm_knowledge", "data_analyzer", "visualizer"]


class ValidatorOutput(BaseModel):
    score: float
    is_valid: bool
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_memory_section(state: AgentState) -> str:
    mem = state.get("user_memory") or "No memory available."
    return USER_MEMORY_SECTION.format(user_memory=mem)


def _get_llm(provider: Optional[str], purpose: str):
    return LLMFactory.create_client(purpose=purpose, provider=provider).get_llm()


def _get_structured_llm(provider: Optional[str], purpose: str, schema):
    return LLMFactory.create_client(purpose=purpose, provider=provider).get_structed_llm(schema)


def _provider(state: AgentState) -> Optional[str]:
    return state.get("llm_provider") or settings.graph_provider


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def context_compressor(state: AgentState) -> dict:
    """Creates a brief summary of the input document if it hasn't been summarized yet."""
    # 1. Skip if summary already exists
    if state.get("content_summary"): 
        return {}
        
    # 2. Get input data
    input_data = (state.get("input_data") or "").strip()
    
    # 3. Skip if NO input data to summarize
    if not input_data: 
        return {"content_summary": ""}

    try:
        llm = _get_llm(_provider(state), "summary")
        response = llm.invoke(CONTEXT_COMPRESSION_PROMPT.format(input_data=input_data[:5000]))
        return {"content_summary": response.content.strip()}
    except Exception as e:
        logger.error(f"[context_compressor] Error: {e}")
        return {"content_summary": ""}


def ambiguity_checker(state: AgentState) -> dict:
    provider = _provider(state)
    prompt = AMBIGUITY_CHECK_PROMPT.format(
        question=state.get("question") or "",
        content_summary=state.get("content_summary") or "N/A",
        user_memory_section=_build_memory_section(state),
        dataframe_head=state.get("dataframe_head") or "N/A",
        dataframe_info=state.get("dataframe_info") or "N/A",
    )
    try:
        res: AmbiguityCheckOutput = _get_structured_llm(provider, "classifier", AmbiguityCheckOutput).invoke(prompt)
        return {"is_ambiguous": res.is_ambiguous, "rejection_reason": res.reason}
    except Exception:
        return {"is_ambiguous": False}


def rejection_handler(state: AgentState) -> dict:
    reason = state.get("rejection_reason") or REJECTION_FALLBACK_ANSWER
    return {"final_answer": reason, "messages": [AIMessage(content=reason)]}


def planner(state: AgentState) -> dict:
    prompt = PLANNER_PROMPT.format(
        question=state.get("question") or "",
        dataframe_head=state.get("dataframe_head") or "N/A",
        dataframe_info=state.get("dataframe_info") or "N/A",
        user_memory_section=_build_memory_section(state),
        data_mode=state.get("data_mode") or "None",
    )
    try:
        res: PlannerOutput = _get_structured_llm(_provider(state), "rag", PlannerOutput).invoke(prompt)
        # Extract tasks from res.sub_tasks and populate state with default status
        sub_tasks = [{"task_type": t.task_type, "description": t.description, "status": "pending", "result": None} for t in res.sub_tasks]
        return {"sub_tasks": sub_tasks}
    except Exception:
        return {"sub_tasks": [{"task_type": "rag", "description": state.get("question", ""), "status": "pending", "result": None}]}


def knowledge_router(state: AgentState) -> dict:
    provider = _provider(state)
    current_task = state.get("current_task")
    if not current_task: return {"route": "llm_knowledge"}
    prompt = KNOWLEDGE_ROUTER_PROMPT.format(
        current_task=current_task["description"],
        content_summary=state.get("content_summary") or "N/A",
        dataframe_head=state.get("dataframe_head") or "N/A",
        dataframe_info=state.get("dataframe_info") or "N/A",
    )
    try:
        res: KnowledgeRouterOutput = _get_structured_llm(provider, "classifier", KnowledgeRouterOutput).invoke(prompt)
        route = res.route
        
        # --- Filter route by data_mode ---
        data_mode = state.get("data_mode")
        if data_mode == "document":
            allowed = {"rag", "web", "llm_knowledge"}
        elif data_mode == "tabular":
            allowed = {"data_analyzer", "visualizer", "llm_knowledge"}
        else:  # None
            allowed = {"llm_knowledge"}

        if route not in allowed:
            # Fallback based on data_mode
            if data_mode == "document":
                route = "rag"
            elif data_mode == "tabular":
                route = "data_analyzer"
            else:
                route = "llm_knowledge"
            logger.warning(f"[knowledge_router] Route '{res.route}' not allowed for mode '{data_mode}'. Fallback -> '{route}'")
            
        return {"route": route}
    except Exception as e:
        logger.error(f"[knowledge_router] Error: {e}")
        return {"route": "llm_knowledge"}


def hyde(state: AgentState) -> dict:
    """Generates a hypothetical answer to improve semantic search."""
    current_task = state.get("current_task")
    if not current_task: return {}
    try:
        prompt = HYDE_PROMPT.format(current_task=current_task["description"])
        res = _get_llm(_provider(state), "rag").invoke(prompt)
        return {"hyde_query": res.content.strip()}
    except Exception:
        return {"hyde_query": current_task["description"]}


def llm_node(state: AgentState) -> dict:
    """Answers using LLM knowledge, with specialized prompts for document/tabular/base modes."""
    provider = _provider(state)
    current_task = state.get("current_task")
    if not current_task: return {}
    
    data_mode = state.get("data_mode")
    
    # 1. Select the appropriate prompt template based on data_mode
    if data_mode == "document":
        template = LLM_KNOWLEDGE_DOC_PROMPT
    elif data_mode == "tabular":
        template = LLM_KNOWLEDGE_TAB_PROMPT
    else:
        template = LLM_KNOWLEDGE_BASE_PROMPT
        
    prompt = template.format(
        user_memory_section=_build_memory_section(state),
        content_summary=state.get("content_summary") or "N/A",
        dataframe_head=state.get("dataframe_head") or "N/A",
        current_task=current_task["description"]
    )
    
    try:
        res = _get_llm(provider, "rag").invoke(prompt)
        result = TaskResult(
            task=current_task["description"],
            type="text",
            content=res.content.strip()
        )
        return {"sub_task_results": [result]}
    except Exception as e:
        logger.error(f"[llm_node] Error: {e}")
        return {"sub_task_results": [TaskResult(task=current_task["description"], type="error", content=str(e))]}



# ---------------------------------------------------------------------------
# Node 7 — Synthesizer (REFACTORED: Heavy Reasoning Node)
# ---------------------------------------------------------------------------

def synthesizer(state: AgentState) -> dict:
    """Consolidation node that performs all cross-task reasoning and formatting."""
    provider = _provider(state)
    results = state.get("sub_task_results") or []
    
    # Format sub-task results for the LLM
    formatted_results = []
    for i, res in enumerate(results):
        task_desc = res.get("task", "Unknown Task")
        r_type = res.get("type", "unknown").upper()
        content = res.get("content", "")
        formatted_results.append(f"### Result {i+1} [{r_type}]\n**Task**: {task_desc}\n**Content**: {content}")
    
    context_str = "\n\n".join(formatted_results) if formatted_results else "(No raw results from tools available)"

    # Identify and separate charts for UI usage vs prompt injection
    all_chart_paths = [r.get("content") for r in results if r.get("type") == "chart"]
    
    prompt = SYNTHESIZER_PROMPT.format(
        question=state.get("question") or "",
        all_context=context_str,
        user_memory_section=_build_memory_section(state),
    )

    try:
        response = _get_llm(provider, "summary").invoke(prompt)
        final_answer = response.content.strip()
    except Exception as e:
        logger.error(f"[synthesizer] Error: {e}")
        final_answer = TECHNICAL_ERROR_RESPONSE.format(error=str(e))

    return {
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
        "chart_paths": all_chart_paths # Persist paths for API response
    }
