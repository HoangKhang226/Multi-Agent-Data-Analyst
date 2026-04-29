"""Mem0-based long-term user memory backed by local Ollama or Gemini models.

Provides a LongTermMemoryManager class for managing user facts and conversation turns.
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional

from pydantic import BaseModel, Field

from src.core.config import settings
from src.utils.logger import logger
from src.llm.factory import LLMFactory
from src.prompt.template import PROMPT_EXTRACT_MEMORY

class FactMetadata(BaseModel):
    type: Literal["preference", "habit", "insight", "skill", "goal", "identity"]
    confidence: Optional[float] = 1.0

class Fact(BaseModel):
    content: str = Field(..., min_length=5)
    metadata: FactMetadata

class FactList(BaseModel):
    facts: List[Fact]

class LongTermMemoryManager:
    """Manager for Mem0-based long-term user memory."""

    def __init__(self):
        self._mem0_instances: Dict[str, Any] = {}
        self._init_lock = asyncio.Lock()

    def _build_mem0_config(self, provider: Optional[str] = None) -> dict:
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

    def get_mem0(self, provider: Optional[str] = None):
        """Return a Memory instance for the given provider, creating it on first call."""
        target_provider = provider or settings.memory_provider
        
        if target_provider not in self._mem0_instances:
            from mem0 import Memory  # lazy import
            config = self._build_mem0_config(provider=target_provider)
            logger.info(f"Initializing Mem0 Memory with {target_provider} backend...")
            self._mem0_instances[target_provider] = Memory.from_config(config)
            logger.info(f"Mem0 Memory for {target_provider} initialized successfully.")
        
        return self._mem0_instances[target_provider]

    async def extract_facts(self, question: str, answer: str, provider: Optional[str] = None) -> List[Fact]:
        """Use the LLM to extract only the most important facts from a Q&A pair."""
        try:
            mem_provider = provider or settings.memory_provider
            llm = LLMFactory.create_client(purpose="summary", provider=mem_provider).get_structed_llm(FactList)
            prompt = PROMPT_EXTRACT_MEMORY.format(question=question, answer=answer)
            response = await llm.ainvoke(prompt)
            
            logger.debug(f"[Memory] Extraction response type: {type(response)}")
            
            facts_list = []
            if isinstance(response, FactList):
                facts_list = response.facts
            elif isinstance(response, dict):
                facts_data = response.get("facts", [])
                for f_dict in facts_data:
                    try:
                        facts_list.append(Fact(**f_dict))
                    except:
                        continue
            
            facts = [
                f for f in facts_list
                if f.content and len(f.content.strip()) > 5
            ]
            return facts
        except Exception as exc:
            logger.warning(f"[Memory] Failed to extract memory: {exc}")
            logger.debug(traceback.format_exc())
            return []

    async def get_user_facts(self, query: str, user_id: str, provider: Optional[str] = None) -> str:
        """Search Mem0 for facts about *user_id* relevant to *query*."""
        if not query or not user_id:
            return ""

        try:
            m = self.get_mem0(provider=provider)
            results = await asyncio.to_thread(
                m.search, query=query, filters={"user_id": user_id}, limit=5
            )

            if not results or not results.get("results"):
                return ""

            facts = [r.get("memory", "") for r in results["results"] if r.get("memory")]
            facts = facts[:5]  # Strictly limit to 5 facts as requested by user
            if not facts:
                return ""

            formatted = "\n".join(f"- {fact}" for fact in facts)
            logger.info(f"[Memory] Retrieved {len(facts)} fact(s) for user '{user_id}'.")
            return formatted

        except Exception as exc:
            logger.warning(f"[Memory] Failed to retrieve facts for '{user_id}': {exc}")
            return ""

    async def save_user_facts(
        self, messages: List[Dict[str, str]], user_id: str, provider: Optional[str] = None
    ) -> List[str]:
        """Persist a conversation turn to Mem0 under *user_id*."""
        if not messages or not user_id or len(messages) < 2:
            return []

        try:
            m = self.get_mem0(provider=provider)
            question = messages[-2]["content"]
            answer = messages[-1]["content"]

            facts = await self.extract_facts(question, answer, provider=provider)
            saved_contents = []
            if facts:
                for fact in facts:
                    meta_dict = fact.metadata.dict()
                    cleaned_meta = {k: v for k, v in meta_dict.items() if v is not None}
                    cleaned_meta["fact_type"] = cleaned_meta.pop("type", "fact")
                    cleaned_meta["user_id"] = user_id
                    
                    await asyncio.to_thread(
                        m.add,
                        messages=[{"role": "user", "content": fact.content}],
                        user_id=user_id,
                        metadata=cleaned_meta
                    )
                    saved_contents.append(fact.content)
            else:
                logger.info(f"[Memory] No facts extracted for user '{user_id}'.")
            
            logger.info(f"[Memory] Saved {len(saved_contents)} conversation fact(s) for user '{user_id}'.")
            return saved_contents
        except Exception as exc:
            logger.warning(f"[Memory] Failed to save facts for '{user_id}': {exc}")
            return []

# Singleton instance
memory_manager = LongTermMemoryManager()

# Module-level aliases for backward compatibility and convenience
get_user_facts = memory_manager.get_user_facts
save_user_facts = memory_manager.save_user_facts
_get_memory = memory_manager.get_mem0