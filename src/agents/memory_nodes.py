"""LangGraph nodes for long-term user memory retrieval and persistence.

These nodes bookend the main pipeline:
  START → retrieve_memory_node → [pipeline] → update_memory_node → END

Both are fully fail-safe — memory errors are logged and the graph continues
as if no memory existed.
"""

from __future__ import annotations

from src.agents.state import AgentState
from src.memory.long_term import get_user_facts, save_user_facts
from src.utils.logger import logger


async def retrieve_memory_node(state: AgentState) -> dict:
    """Fetch relevant user facts from Mem0 and inject into state.

    Reads:  question, user_id
    Writes: user_memory
    """
    question = state.get("question", "")
    user_id = state.get("user_id", "guest")
    logger.debug(f"[DEBUG] retrieve_memory_node: user_id={user_id}, state={list(state.keys())}")

    logger.info(f"[MemoryNode] Retrieving memory for user '{user_id}'...")

    user_memory = await get_user_facts(query=question, user_id=user_id)

    if user_memory:
        logger.info(f"[MemoryNode] Found memory context for '{user_id}'.")
    else:
        logger.info(f"[MemoryNode] No prior memory for '{user_id}'.")

    return {"user_memory": user_memory or ""}


async def update_memory_node(state: AgentState) -> dict:
    """Persist the current conversation turn to Mem0.

    Reads:  question, final_answer, user_id
    Writes: (nothing — side-effect only)
    """
    user_id = state.get("user_id", "guest")
    question = state.get("question", "")
    final_answer = state.get("final_answer", "")
    logger.debug(f"[DEBUG] update_memory_node: user_id={user_id}, state={list(state.keys())}")

    if not question or not final_answer:
        logger.info("[MemoryNode] Skipping memory update — no Q/A pair.")
        return {}

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": final_answer},
    ]

    logger.info(f"[MemoryNode] Saving conversation turn for user '{user_id}'...")
    await save_user_facts(messages=messages, user_id=user_id)
    return {}