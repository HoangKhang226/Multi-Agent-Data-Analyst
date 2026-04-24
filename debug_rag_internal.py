import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.retrieval.vector_db import VectorDBManager
from src.llm.embeddings import EmbeddingFactory
from src.core.config import settings
from llama_index.core.schema import TextNode

def test_rag_quality():
    print("=== Deep RAG Quality Test ===")
    
    # 1. Setup
    provider = "gemini"
    collection = "test_rag_quality"
    factory = EmbeddingFactory()
    embed_model = factory.get_embedding(provider=provider)
    db = VectorDBManager(embed_model, provider=provider)
    
    # 2. Reset and Add controlled nodes
    print(f"\n[Step 1] Adding controlled nodes to collection '{collection}'...")
    nodes = [
        TextNode(text="The secret password is 'ANTIGRAVITY-2026'.", id_="node_1"),
        TextNode(text="The capital of Vietnam is Hanoi.", id_="node_2"),
        TextNode(text="Python is a popular programming language.", id_="node_3")
    ]
    db.add_documents(nodes, collection_name=collection)
    
    # 3. Verify Summary saving
    print("\n[Step 2] Saving and checking summary...")
    summary_text = "This document contains a secret password and facts about Vietnam and Python."
    db.save_summary(collection, summary_text)
    saved_summary = db.get_summary(collection)
    print(f"   Saved Summary: {saved_summary}")
    
    # 4. Test Retrieval
    print("\n[Step 3] Testing Context Retrieval...")
    retriever = db.get_retriever(similarity_top_k=2, collection_name=collection)
    if not retriever:
        print("❌ Failed to get retriever.")
        return

    query = "What is the secret password?"
    results = retriever.retrieve(query)
    
    print(f"   Query: {query}")
    print(f"   Found {len(results)} nodes.")
    for i, res in enumerate(results):
        print(f"   Node {i+1}: {res.node.get_content()[:100]}... (Score: {res.score if hasattr(res, 'score') else 'N/A'})")
    
    # 5. Check Directory Structure
    print("\n[Step 4] Verifying File Structure...")
    pdir = db.get_persist_dir(collection)
    print(f"   Isolated Dir: {pdir}")
    if (pdir / "docstore.json").exists():
        print("   ✅ docstore.json found in isolated directory.")
    else:
        print("   ❌ docstore.json MISSING from isolated directory.")

if __name__ == "__main__":
    test_rag_quality()
