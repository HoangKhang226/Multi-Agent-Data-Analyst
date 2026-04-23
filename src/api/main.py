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
from typing import List, Optional

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
from src.memory.long_term import _get_memory

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
app.mount("/charts", StaticFiles(directory="output_charts"), name="charts") # biến folder output_charts thành public URL /charts


@app.on_event("startup") # chạy khi app khởi động để tránh gọi llm nhiều lần
async def on_startup():
    """Configure LlamaIndex global settings once on startup."""
    try:
        LLMFactory.configure_llama_index_settings()
        logger.info("[startup] LlamaIndex global settings configured.")
    except Exception as e:
        logger.error(f"[startup] Failed to configure LlamaIndex settings: {e}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = "guest"
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    # Optional pre-loaded summary (e.g. from Streamlit session)
    content_summary: Optional[str] = ""
    dataframe_head: Optional[str] = ""
    dataframe_info: Optional[str] = ""


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
        "message": "Chat With Data API is running ✅",
        "project": settings.app.project_name,
        "version": settings.app.version,
        "default_provider": settings.llm.provider,
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
    allowed_docs = {".pdf", ".docx"} # các định dạng file được phép
    allowed_tables = {".csv", ".xlsx", ".xls"} # các định dạng file được phép\
    result=[]
    for file in files:
        suffix = Path(file.filename).suffix.lower() # suffix là đuôi file và chuyển sang chữ thường
        
        if suffix not in allowed_docs and suffix not in allowed_tables: # nếu đuôi file không thuộc allowed_docs và allowed_tables
            raise HTTPException(
                status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(allowed_docs | allowed_tables)}",
        )

        logger.info(f"[/ingest] Ingesting file: {file.filename}")

        # Save to a temp file
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp: # tạo file tạm để lưu file upload
            shutil.copyfileobj(file.file, tmp) # copy file upload vào file tạm
            tmp_path = Path(tmp.name) # lấy đường dẫn file tạm
            
        try:
            if suffix in allowed_docs: # nếu đuôi file thuộc allowed_docs
                orchestrator = IngestionOrchestrator(provider=embedding_provider)
                res = await orchestrator.ingest_pdf(tmp_path, embedding_provider=embedding_provider)
                result.append(IngestResponse(**res))
            else:
                # Tabular data handling
                orchestrator = IngestionOrchestrator(provider="data_analyzer")
                res = await orchestrator.ingest_tabular(tmp_path)
                
                df = res.pop("df")
                # Persist DF for pandas_runner (MVP: single active file)
                storage_dir = Path("storage")
                storage_dir.mkdir(exist_ok=True)
                active_df_path = storage_dir / f"{file.filename}.parquet"
                
                df.to_parquet(active_df_path)
                logger.info(f"[/ingest] Persisted tabular data to {active_df_path}")
                
                # Construct a response compatible with IngestResponse
                result.append(IngestResponse(
                    status="success",
                    filename=file.filename,
                    chunks_count=0,  # No vector chunks for tabular
                    collection="tabular",
                    summary=res["dataframe_head"],
                    info=res.get("dataframe_info", ""),
                    provider="data_analyzer",
                ))
        except Exception as e:
            logger.error(f"[/ingest] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return result


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Submit a question and receive an answer from the LangGraph agent pipeline.
    
    The pipeline includes Mem0 memory retrieval/update for personalised conversations.
    Pass `user_id` to enable per-user long-term memory.
    """
    llm_provider = request.llm_provider or settings.llm.provider
    embedding_provider = request.embedding_provider or settings.llm.provider
    user_id = request.user_id or "guest"

    # Hot-swap LlamaIndex settings
    LLMFactory.configure_llama_index_settings(provider=llm_provider)

    # Load summary if not provided
    content_summary = request.content_summary or ""
    if not content_summary:
        try:
            embedding = EmbeddingFactory().get_embedding(provider=embedding_provider)
            db = VectorDBManager(embedding_model=embedding, provider=embedding_provider)
            content_summary = db.get_summary(settings.storage.collection_name) or ""
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
        collection_name=settings.storage.collection_name,
        user_id=user_id,
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

    try:
        result = await agent_graph.ainvoke(state)
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
        logger.error(f"[/chat] Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Memory Endpoints
# ---------------------------------------------------------------------------


@app.get("/memory/{user_id}", response_model=MemoryResponse, tags=["Memory"])
def get_memory(user_id: str):
    """Retrieve all stored long-term memories for a specific user."""
    try:
        mem0 = _get_memory()
        results = mem0.get_all(user_id=user_id)
        # Handle both list and dict results from mem0
        memories = results.get("results", []) if isinstance(results, dict) else results
        return MemoryResponse(user_id=user_id, memories=memories or [])
    except Exception as e:
        logger.error(f"[get_memory] Error: {e}")
        return MemoryResponse(user_id=user_id, memories=[])


@app.delete("/memory/{user_id}", tags=["Memory"])
def clear_memory(user_id: str):
    """Clear all stored long-term memories for a specific user."""
    try:
        mem0 = _get_memory()
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
    provider = embedding_provider or settings.llm.provider
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
