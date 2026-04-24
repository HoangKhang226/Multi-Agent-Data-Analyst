import asyncio
import os
from dotenv import load_dotenv

# Mocking state and internal imports
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.retrieval.vector_db import VectorDBManager
from src.llm.embeddings import EmbeddingFactory
from llama_index.core import Settings

async def debug_retrieval_quality():
    print("=== Debugging RAG Retrieval Quality ===")
    
    provider = "gemini"
    collection_name = "chat_with_data"
    query = "các mô hình machine learning được tài liệu đưa ra là gì"
    
    embed_factory = EmbeddingFactory()
    embedding_model = embed_factory.get_embedding(provider=provider)
    
    # 1. Test basic vector retrieval
    db = VectorDBManager(embedding_model=embedding_model, provider=provider)
    index = db.get_index(collection_name)
    
    if not index:
        print(f"❌ Cannot load index for {collection_name}")
        return

    print(f"\n[Test 1] Basic Vector Search (Top-k: 5)")
    base_retriever = index.as_retriever(similarity_top_k=5)
    nodes = base_retriever.retrieve(query)
    print(f" -> Found {len(nodes)} nodes.")
    for i, n in enumerate(nodes):
        score = n.get_score()
        print(f"    Node {i} | Score: {score:.4f} | Content: {n.get_content()[:100]}...")

    # 2. Test AutoMergingRetriever
    print(f"\n[Test 2] AutoMergingRetriever")
    from llama_index.core.retrievers import AutoMergingRetriever
    merging_retriever = AutoMergingRetriever(base_retriever, index.storage_context, verbose=True)
    merged_nodes = merging_retriever.retrieve(query)
    print(f" -> Found {len(merged_nodes)} merged nodes.")
    for i, n in enumerate(merged_nodes):
        print(f"    Merged Node {i} | Content: {n.get_content()[:100]}...")

    # 3. Test Hybrid Search
    print(f"\n[Test 3] Hybrid Search (Vector + BM25)")
    hybrid_retriever = db.get_hybrid_retriever(similarity_top_k=5, collection_name=collection_name)
    if hybrid_retriever:
        hybrid_nodes = hybrid_retriever.retrieve(query)
        print(f" -> Found {len(hybrid_nodes)} hybrid nodes.")
        for i, n in enumerate(hybrid_nodes):
            print(f"    Hybrid Node {i} | Content: {n.get_content()[:100]}...")
    else:
        print(" -> Hybrid retriever not available (possibly no nodes for BM25)")

if __name__ == "__main__":
    asyncio.run(debug_retrieval_quality())
