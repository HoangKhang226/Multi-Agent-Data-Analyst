"""Mem0-based long-term user memory backed by local Ollama models.

Provides two async helpers consumed by LangGraph memory nodes:
  - get_user_facts()  — search past memories relevant to the current query
  - save_user_facts() — persist the latest conversation turn

All operations are fail-safe: if Ollama or Mem0 is unavailable the caller
receives empty results and the pipeline continues without memory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Dict

from src.core.config import settings
from src.utils.logger import logger

# Lazy import to avoid import-time crash when mem0ai is not installed
_mem0_instance = None
_init_lock = asyncio.Lock()


def _build_mem0_config() -> dict:
    """Build the Mem0 configuration dict using Ollama for both LLM and embedder."""
    ollama_base = settings.ollama.base_url  # e.g. "http://localhost:11434"
    llm_model = settings.ollama.rag_model   # e.g. "qwen3:8b"
    embed_model = settings.ollama.embed_model # e.g. "nomic-embed-text"

    chroma_path = str(Path(settings.memory.chroma_path).resolve())
    history_db = str(Path(settings.memory.history_db_path).resolve())

    return {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "ollama_base_url": ollama_base,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embed_model,
                "ollama_base_url": ollama_base,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_user_memory",
                "path": chroma_path,
            },
        },
        "history_db_path": history_db,
        "version": "v1.1",
    }


def _get_memory():
    """Return the singleton Memory instance, creating it on first call."""
    global _mem0_instance
    if _mem0_instance is None:
        from mem0 import Memory  # lazy import

        config = _build_mem0_config()
        logger.info("Initializing Mem0 Memory with Ollama backend...")
        _mem0_instance = Memory.from_config(config)
        logger.info("Mem0 Memory initialized successfully.")
    return _mem0_instance


async def get_user_facts(query: str, user_id: str) -> str:
    """Search Mem0 for facts about *user_id* relevant to *query*.

    Returns a formatted multi-line string, or "" if nothing is found or
    an error occurs.
    """
    if not query or not user_id:
        return ""

    try:
        m = _get_memory()
        # mem0 search is synchronous — run in a thread to keep the event loop free
        results = await asyncio.to_thread(
            m.search, query=query, filters={"user_id": user_id}, limit=5
        )

        if not results or not results.get("results"):
            return ""

        facts = [r.get("memory", "") for r in results["results"] if r.get("memory")]
        if not facts:
            return ""

        formatted = "\n".join(f"- {fact}" for fact in facts)
        logger.info(
            f"[Memory] Retrieved {len(facts)} fact(s) for user '{user_id}'."
        )
        return formatted

    except Exception as exc:
        logger.warning(f"[Memory] Failed to retrieve facts for '{user_id}': {exc}")
        return ""


async def save_user_facts(
    messages: List[Dict[str, str]], user_id: str
) -> None:
    """Persist a conversation turn to Mem0 under *user_id*.

    *messages* should follow the OpenAI chat format::

        [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
        ]

    Errors are logged and silently swallowed so the pipeline is never blocked.
    """
    if not messages or not user_id:
        return

    try:
        m = _get_memory()
        await asyncio.to_thread(m.add, messages=messages, user_id=user_id)
        logger.info(f"[Memory] Saved conversation turn for user '{user_id}'.")
    except Exception as exc:
        logger.warning(f"[Memory] Failed to save facts for '{user_id}': {exc}")