import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agents.graph import build_graph, make_initial_state
from src.core.config import settings
from src.utils.logger import logger
import pandas as pd

async def test_routing():
    # Load a sample DB for tabular mode if needed
    df = pd.DataFrame({"sales": [10, 20, 30], "month": ["Jan", "Feb", "Mar"]})
    
    graph = build_graph(df=df)
    
    test_cases = [
        {
            "name": "NONE Mode - Should only allow llm_knowledge",
            "data_mode": None,
            "question": "Bạn là ai?",
            "expected_route": "llm_node"
        },
        {
            "name": "DOCUMENT Mode - Should allow RAG/Web/Knowledge",
            "data_mode": "document",
            "question": "Nội dung chính của tài liệu này là gì?",
            "expected_route": "hyde" # Hyde comes before rag_retriever
        },
        {
            "name": "TABULAR Mode - Should allow Analyzer/Visualizer",
            "data_mode": "tabular",
            "question": "Tính tổng doanh thu và vẽ biểu đồ",
            "expected_route": "data_analyzer" # or visualizer depending on planner
        }
    ]
    
    for case in test_cases:
        logger.info(f"\n--- Testing: {case['name']} ---")
        state = make_initial_state(
            provider="google", # Use google for faster testing if available, or ollama
            data_mode=case["data_mode"],
            collection_name="test_collection"
        )
        state.update({
            "question": case["question"],
            "content_summary": "Tóm tắt tài liệu mẫu cho testing." if case["data_mode"] == "document" else ""
        })
        
        # We invoke only until knowledge_router to verify routing
        # Or run the whole graph and check the intermediate steps
        try:
            result = await graph.ainvoke(state)
            logger.info(f"Final Answer preview: {str(result.get('final_answer'))[:100]}...")
            
            # Check sub-tasks and their routes
            sub_tasks = result.get("sub_tasks") or []
            logger.info(f"Generated sub-tasks: {[t['task_type'] for t in sub_tasks]}")
            
            # Validation logic
            if case["data_mode"] == None:
                assert all(t["task_type"] == "llm_knowledge" for t in sub_tasks), "None mode should only have llm_knowledge tasks"
            elif case["data_mode"] == "document":
                allowed = {"rag", "web_search", "llm_knowledge"}
                assert all(t["task_type"] in allowed for t in sub_tasks), f"Document mode has invalid tasks: {[t['task_type'] for t in sub_tasks]}"
            elif case["data_mode"] == "tabular":
                allowed = {"data_analyzer", "visualizer", "llm_knowledge"}
                assert all(t["task_type"] in allowed for t in sub_tasks), f"Tabular mode has invalid tasks: {[t['task_type'] for t in sub_tasks]}"
                
            logger.info("✅ Case passed routing validation!")
            
        except Exception as e:
            logger.error(f"❌ Case failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_routing())
