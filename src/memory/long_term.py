"""Mem0-based long-term user memory backed by local Ollama models.

Provides two async helpers consumed by LangGraph memory nodes:
  - get_user_facts()  — search past memories relevant to the current query
  - save_user_facts() — persist the latest conversation turn

All operations are fail-safe: if Ollama or Mem0 is unavailable the caller
receives empty results and the pipeline continues without memory.
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import List, Dict, Any

from src.core.config import settings
from src.utils.logger import logger
from src.llm.factory import LLMFactory
from src.prompt.template import PROMPT_EXTRACT_MEMORY
from pydantic import BaseModel, Field
from typing import Literal, Optional
# Lazy import to avoid import-time crash when mem0ai is not installed
# Cache for Memory instances by provider name
_mem0_instances: Dict[str, Any] = {}
_init_lock = asyncio.Lock()
class FactMetadata(BaseModel):
    type: Literal["preference", "habit", "insight", "skill", "goal"]
    confidence: Optional[float] = 1.0

class Fact(BaseModel):
    content: str = Field(..., min_length=5)
    metadata: FactMetadata

class FactList(BaseModel):
    facts: List[Fact]

def _build_mem0_config(provider: Optional[str] = None) -> dict:
    """Build the Mem0 configuration dict for the specified or configured provider."""
    ollama_base = settings.ollama.base_url
    target_provider = provider or settings.memory_provider

    if target_provider == "ollama":
        llm_model = settings.ollama.model
        embed_model = settings.ollama.embed_model
    else:
        # Default to Gemini
        llm_model = settings.gemini.model
        embed_model = settings.embedding.google

    chroma_path = str(Path(settings.memory.chroma_path).resolve())
    history_db = str(Path(settings.memory.history_db_path).resolve())

    llm_config = {
        "model": llm_model,
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    if target_provider == "ollama":
        llm_config["ollama_base_url"] = ollama_base
    elif target_provider == "gemini":
        llm_config["api_key"] = settings.google_api_key

    embedder_config = {
        "model": embed_model,
    }
    if target_provider == "ollama":
        embedder_config["ollama_base_url"] = ollama_base
    elif target_provider == "gemini":
        embedder_config["api_key"] = settings.google_api_key

    return {
        "llm": {
            "provider": target_provider,
            "config": llm_config,
        },
        "embedder": {
            "provider": target_provider,
            "config": embedder_config,
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


def _get_memory(provider: Optional[str] = None):
    """Return a Memory instance for the given provider, creating it on first call."""
    global _mem0_instances
    target_provider = provider or settings.memory_provider
    
    if target_provider not in _mem0_instances:
        from mem0 import Memory  # lazy import

        # Temporarily override settings if needed, or just pass to build_config
        config = _build_mem0_config(provider=target_provider)
        logger.info(f"Initializing Mem0 Memory with {target_provider} backend...")
        _mem0_instances[target_provider] = Memory.from_config(config)
        logger.info(f"Mem0 Memory for {target_provider} initialized successfully.")
    
    return _mem0_instances[target_provider]

async def extract_memory(question: str, answer: str, provider: Optional[str] = None) -> List[Fact]:
    """Use the LLM to extract only the most important facts from a Q&A pair.

    Returns a list of facts, or [] if extraction fails.
    """
    try:
        mem_provider = provider or settings.memory_provider
        llm = LLMFactory.create_client(purpose="summary", provider=mem_provider).get_structed_llm(FactList)
        prompt = PROMPT_EXTRACT_MEMORY.format(question=question, answer=answer)
        response = await llm.ainvoke(prompt)
        
        logger.debug(f"[Memory] Extraction response type: {type(response)}")
        
        # Handle different response types from with_structured_output
        facts_list = []
        if isinstance(response, FactList):
            facts_list = response.facts
        elif isinstance(response, dict):
            # Sometimes it returns a dict even with Pydantic class
            facts_data = response.get("facts", [])
            for f_dict in facts_data:
                try:
                    facts_list.append(Fact(**f_dict))
                except:
                    continue
        
        facts = [
            f for f in facts_list
            if f.content and len(f.content.strip()) > 10
        ]
        if not facts:
            return []
        return facts
    except Exception as exc:
        logger.warning(f"[Memory] Failed to extract memory: {exc}")
        logger.debug(traceback.format_exc())
        return []

async def get_user_facts(query: str, user_id: str, provider: Optional[str] = None) -> str:
    """Search Mem0 for facts about *user_id* relevant to *query*.

    Returns a formatted multi-line string, or "" if nothing is found or
    an error occurs.
    """
    if not query or not user_id:
        return ""

    try:
        m = _get_memory(provider=provider)
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
    messages: List[Dict[str, str]], user_id: str, provider: Optional[str] = None
) -> List[str]:
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

    if len(messages) < 2:
        return
    try:
        m = _get_memory(provider=provider) # instance of mem0

        question = messages[-2]["content"]
        answer = messages[-1]["content"]

        # extract facts from the conversation turn
        facts = await extract_memory(question, answer, provider=provider)
        # save facts to memory
        saved_contents = []
        if facts:
            for fact in facts:
                await asyncio.to_thread(m.add, # Async function to add messages to memory
                messages=[{"role": "user", "content": fact.content}],
                user_id=user_id,
                metadata={
                        **fact.metadata.dict(),
                        "type": "fact", 
                        "user_id": user_id
                        }
                )
                saved_contents.append(fact.content)
        else:
            logger.info(f"[Memory] No facts extracted from conversation turn for user '{user_id}'.")
        
        logger.info(f"[Memory] Saved {len(saved_contents)} conversation fact(s) for user '{user_id}'.")
        return saved_contents
    except Exception as exc:
        logger.warning(f"[Memory] Failed to save facts for '{user_id}': {exc}")
        return []