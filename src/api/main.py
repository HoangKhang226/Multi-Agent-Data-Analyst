"""FastAPI backend for the Chat with Data — Hierarchical RAG system.

Endpoints
---------
GET  /           — API info
GET  /health     — Health check
POST /ingest     — Upload PDF/DOCX/CSV/Excel and index into vector DB
POST /chat       — Send question through the LangGraph agent pipeline
GET  /memory/{user_id}    — Retrieve stored memories for a user
DELETE /memory/{user_id}  — Clear stored memories for a user
DELETE /reset    — Clear vector DB for a given provider
"""

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.config import settings
from src.utils.logger import logger
from src.core.orchestrator import IngestionOrchestrator
from src.llm.embeddings import EmbeddingFactory
from src.retrieval.vector_db import VectorDBManager
from src.agents.graph import build_graph, make_initial_state
from src.llm.factory import LLMFactory
from src.memory.long_term import memory_manager
from src.db.database import init_db, get_db, db_manager
from src.db.crud import create_dataset, create_session, log_message

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chat With Data — Hierarchical RAG API",
    description=(
        "Agentic RAG system: upload documents (PDF/CSV/Excel), chat with your data. "
        "Supports Gemini and Ollama backends. Includes Mem0 long-term user memory."
    ),
    version="1.1.0",
)

# Serve charts for UI
charts_dir = Path("output_charts")
charts_dir.mkdir(exist_ok=True)
app.mount("/charts", StaticFiles(directory="output_charts"), name="charts") # Map output_charts folder to public URL /charts


@app.on_event("startup") # Run on startup to avoid multiple LLM re-configs
async def on_startup():
    """Configure LlamaIndex global settings once on startup and initialize DB."""
    try:
        init_db()
        LLMFactory.configure_llama_index_settings()
        logger.info("[startup] LlamaIndex global settings configured and DB initialized.")
    except Exception as e:
        logger.error(f"[startup] Failed to configure settings: {e}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = "guest"
    llm_provider: Optional[str] = None
    memory_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    # Optional pre-loaded summary (e.g. from Streamlit session)
    content_summary: Optional[str] = ""
    dataframe_head: Optional[str] = ""
    dataframe_info: Optional[str] = ""
    
    # NEW: Control routing
    data_mode: Optional[Literal["document", "tabular"]] = None
    retrieval_mode: Optional[Literal["hierarchical", "hybrid"]] = "hierarchical"
    # NEW: Target collection (per-file isolation)
    collection_name: Optional[str] = None

  
class ChatResponse(BaseModel):
    question: str
    answer: Optional[str]
    is_ambiguous: bool
    rejection_reason: Optional[str]
    sub_tasks: list
    chart_paths: List[str] = []
    meta: dict


class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_count: int
    collection: str
    summary: str
    info: Optional[str] = ""
    provider: str
    data_mode: Optional[Literal["document", "tabular"]] = None


class ResetResponse(BaseModel):
    status: str
    message: str


class MemoryResponse(BaseModel):
    user_id: str
    memories: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Chat With Data API is running \u2705",
        "project": settings.app.project_name,
        "version": settings.app.version,
        "default_provider": settings.graph_provider,
    }


@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "healthy", "version": settings.app.version}


@app.post("/ingest", response_model=List[IngestResponse], tags=["Documents"])
async def ingest_document(
    files: List[UploadFile] = File(..., description="Upload one or more documents"),
    embedding_provider: Optional[str] = Query(
        None, description="Embedding backend: 'google' | 'ollama'"
    ),
):
    """Upload a document (PDF, DOCX, CSV, Excel), extract/process, and index."""
    allowed_docs = {".pdf", ".docx"} # Supported document formats
    allowed_tables = {".csv", ".xlsx", ".xls"} # Supported tabular formats
    result=[]
    for file in files:
        suffix = Path(file.filename).suffix.lower() # Normalize file extension
        
        if suffix not in allowed_docs and suffix not in allowed_tables: # Check if format is supported
            raise HTTPException(
                status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(allowed_docs | allowed_tables)}",
        )

        logger.info(f"[/ingest] Ingesting file: {file.filename}")

        # Save to a temp file
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp: # Create temp file for upload
            shutil.copyfileobj(file.file, tmp) # Save uploaded content
            tmp_path = Path(tmp.name) # Get path to temp file
            
        try:
            if suffix in allowed_docs: # Process as PDF/Docx
                orchestrator = IngestionOrchestrator(provider=embedding_provider)
                res = await orchestrator.ingest_pdf(
                    tmp_path, 
                    original_filename=file.filename,
                    embedding_provider=embedding_provider
                )
                res["data_mode"] = "document" # Explicitly set for UI/Routing
                
                # Persist dataset in SQL DB
                try:
                    with db_manager.session() as db_session:
                        create_dataset(
                            db=db_session,
                            user_id="guest",
                            filename=file.filename,
                            collection_name=res["collection"],
                            provider=embedding_provider or "default",
                            data_mode="document",
                            chunks_count=res["chunks_count"],
                            summary=res["summary"]
                        )
                except Exception as db_err:
                    logger.warning(f"[/ingest] Failed to persist dataset record: {db_err}")

                result.append(IngestResponse(**res))
            else:
                # Tabular data handling
                orchestrator = IngestionOrchestrator(provider="data_analyzer")
                res = await orchestrator.ingest_tabular(tmp_path)
                
                df = res.pop("df")
                # Persist DF for pandas_runner (MVP: single active file)
                storage_dir = Path("storage")
                storage_dir.mkdir(exist_ok=True)
                active_df_path = storage_dir / "active_df.parquet"
                
                df.to_parquet(active_df_path)
                logger.info(f"[/ingest] Persisted tabular data to {active_df_path}")
                
                # Use per-file collection name for tabular (parquet path key)
                tabular_collection = IngestionOrchestrator.make_collection_name(file.filename)

                # Persist dataset in SQL DB
                try:
                    with db_manager.session() as db_session:
                        create_dataset(
                            db=db_session,
                            user_id="guest",
                            filename=file.filename,
                            collection_name=tabular_collection,
                            provider="data_analyzer",
                            data_mode="tabular",
                            chunks_count=0,
                            summary=res["dataframe_head"]
                        )
                except Exception as db_err:
                    logger.warning(f"[/ingest] Failed to persist tabular dataset record: {db_err}")


                # Construct a response compatible with IngestResponse
                result.append(IngestResponse(
                    status="success",
                    filename=file.filename,
                    chunks_count=0,  # No vector chunks for tabular
                    collection=tabular_collection,
                    summary=res["dataframe_head"],
                    info=res.get("dataframe_info", ""),
                    provider="data_analyzer",
                    data_mode="tabular",
                ))
        except Exception as e:
            logger.error(f"[/ingest] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return result


@app.get("/collections", tags=["Documents"])
def list_collections(
    embedding_provider: Optional[str] = Query(None, description="Provider: 'google' | 'ollama'")
):
    """List all indexed document collections available for querying."""
    provider = embedding_provider or settings.graph_provider
    try:
        embedding = EmbeddingFactory().get_embedding(provider=provider)
        db = VectorDBManager(embedding_model=embedding, provider=provider)
        summaries = db.get_summary()  # returns full metadata dict
        collections = [
            {"collection_name": k, "summary": v[:200] if isinstance(v, str) else str(v)[:200]}
            for k, v in summaries.items()
        ]
        return {"provider": provider, "collections": collections}
    except Exception as e:
        logger.error(f"[/collections] Error: {e}")
        return {"provider": provider, "collections": []}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Submit a question and receive an answer from the LangGraph agent pipeline.
    
    The pipeline includes Mem0 memory retrieval/update for personalised conversations.
    Pass `user_id` to enable per-user long-term memory.
    """
    llm_provider = request.llm_provider or settings.graph_provider
    memory_provider = request.memory_provider or settings.memory_provider
    embedding_provider = request.embedding_provider or settings.graph_provider
    user_id = request.user_id or "guest"

    # Hot-swap LlamaIndex settings
    LLMFactory.configure_llama_index_settings(provider=llm_provider)

    # Resolve which collection to query
    collection_name = request.collection_name or settings.storage.collection_name

    # Load summary if not provided
    content_summary = request.content_summary or ""
    if not content_summary:
        try:
            embedding = EmbeddingFactory().get_embedding(provider=embedding_provider)
            db = VectorDBManager(embedding_model=embedding, provider=embedding_provider)
            content_summary = db.get_summary(collection_name) or ""
        except Exception:
            pass

    # Load persisted DF if it exists
    df = None
    active_df_path = Path("storage/active_df.parquet")
    if active_df_path.exists():
        import pandas as pd
        try:
            df = pd.read_parquet(active_df_path)
            logger.info("[/chat] Loaded active DataFrame from persistence.")
        except Exception as e:
            logger.warning(f"[/chat] Failed to load active DF: {e}")

    # Build graph with the loaded DataFrame
    agent_graph = build_graph(df=df)

    state = make_initial_state(
        provider=llm_provider,
        embedding_provider=embedding_provider,
        memory_provider=memory_provider,
        collection_name=collection_name,
        user_id=user_id,
        data_mode=request.data_mode,
        retrieval_mode=request.retrieval_mode,
    )
    state.update(
        {
            "question": request.question,
            "content_summary": content_summary,
            "dataframe_head": request.dataframe_head or "",
            "dataframe_info": request.dataframe_info or "",
        }
    )

    logger.info(f"[/chat] question='{request.question}' | llm={llm_provider} | user={user_id}")

    from src.db.crud import get_dataset_by_collection
    db_session = db_manager.get_session()
    try:
        ds = get_dataset_by_collection(db_session, collection_name)
        dataset_id = ds.id if ds else None

        # Create session & log user message
        chat_session = create_session(db_session, user_id=user_id, dataset_id=dataset_id)
        log_message(db_session, chat_session.id, "user", request.question)

        # Inject the session ID into agent state so nodes can log agent runs
        state["session_id"] = chat_session.id

        result = await agent_graph.ainvoke(state)

        final_answer = result.get("final_answer")
        # Log assistant response
        log_message(
            db_session, chat_session.id, "assistant",
            final_answer if final_answer else str(result),
        )
        db_session.commit()

        return ChatResponse(
            question=request.question,
            answer=result.get("final_answer"),
            is_ambiguous=result.get("is_ambiguous", False),
            rejection_reason=result.get("rejection_reason") or None,
            sub_tasks=result.get("sub_tasks", []),
            chart_paths=result.get("chart_paths") or [],
            meta={
                "llm": llm_provider,
                "embedding": embedding_provider,
                "user_id": user_id,
                "engine": "LlamaIndex AutoMerging",
            },
        )
    except Exception as e:
        db_session.rollback()
        logger.error(f"[/chat] Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db_session.close()


# ---------------------------------------------------------------------------
# Memory Endpoints
# ---------------------------------------------------------------------------


@app.get("/memory/{user_id}", response_model=MemoryResponse, tags=["Memory"])
def get_memory(
    user_id: str,
    provider: Optional[str] = Query(None, description="Memory provider: 'gemini' | 'ollama'")
):
    """Retrieve all stored long-term memories for a specific user."""
    try:
        mem0 = memory_manager.get_mem0(provider=provider)
        results = mem0.get_all(filters={"user_id": user_id})
        # Handle both list and dict results from mem0
        memories = results.get("results", []) if isinstance(results, dict) else results
        return MemoryResponse(user_id=user_id, memories=memories or [])
    except Exception as e:
        logger.error(f"[get_memory] Error: {e}")
        return MemoryResponse(user_id=user_id, memories=[])


@app.delete("/memory/{user_id}", tags=["Memory"])
def clear_memory(
    user_id: str,
    provider: Optional[str] = Query(None, description="Memory provider: 'gemini' | 'ollama'")
):
    """Clear all stored long-term memories for a specific user."""
    try:
        mem0 = memory_manager.get_mem0(provider=provider)
        mem0.delete_all(user_id=user_id)
        return {"status": "success", "message": f"All memories for '{user_id}' have been cleared."}
    except Exception as e:
        logger.error(f"[clear_memory] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear user memories: {e}")


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


@app.delete("/reset", response_model=ResetResponse, tags=["Admin"])
async def reset_database(
    embedding_provider: Optional[str] = Query(
        None, description="Provider whose storage to clear: 'google' | 'ollama'"
    ),
):
    """Delete all indexed documents for the specified embedding provider."""
    provider = embedding_provider or settings.graph_provider
    try:
        embedding = EmbeddingFactory().get_embedding(provider=provider)
        db = VectorDBManager(embedding_model=embedding, provider=provider)
        success = db.reset_db()
        if success:
            return ResetResponse(
                status="success",
                message=f"Vector DB for provider '{provider}' has been cleared.",
            )
        raise HTTPException(status_code=500, detail="reset_db() returned False.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/reset] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
