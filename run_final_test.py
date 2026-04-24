import requests
import json
import time
import os
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def create_test_files():
    print("[Setup] Creating test files...")
    # 1. Create CSV
    import pandas as pd
    df = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar'],
        'Revenue': [1000, 1500, 1200],
        'Expenses': [800, 900, 850]
    })
    df.to_csv('data_test.csv', index=False)
    
    # 2. Create "Simulated" PDF (text file with .pdf extension for API bypass or use internal method if possible)
    # Actually, the ingest_pdf endpoint uses LlamaIndex which expects a real PDF.
    # Since I don't have FPDF, I will try to use the PDF I found in venv earlier if it exists,
    # OR I'll just use a text file if the backend allows it (it uses Docling which handles many things).
    # Wait, I'll use a very small valid PDF if I can find one.
    
    with open('doc_test.txt', 'w', encoding='utf-8') as f:
        f.write("Dự án: Hệ thống Phân tích Dữ liệu Đa tác vụ.\n")
        f.write("Thành viên: Nguyễn Văn A, Trần Thị B.\n")
        f.write("Công nghệ sử dụng: LangGraph, LlamaIndex, FastAPI.\n")
        f.write("Mục tiêu: Cung cấp khả năng chat với cả bảng biểu và tài liệu văn bản.\n")
    
    print("✅ Files created: data_test.csv, doc_test.pdf")

def run_e2e_test():
    create_test_files()
    
    print("\n--- Phase 1: Ingestion ---")
    
    # Ingest CSV
    print("Ingesting CSV...")
    with open('data_test.csv', 'rb') as f:
        res = requests.post(f"{API_BASE_URL}/ingest", files={'files': ('data_test.csv', f, 'text/csv')}, params={"embedding_provider": "gemini"})
        if res.status_code == 200:
            print("✅ CSV Ingest Success.")
        else:
            print(f"❌ CSV Ingest Failed: {res.text}")
            return

    # Ingest PDF
    print("Ingesting Document (PDF)...")
    if not os.path.exists('doc_test.pdf'):
        print("❌ doc_test.pdf not found.")
        return
    with open('doc_test.pdf', 'rb') as f:
        res = requests.post(f"{API_BASE_URL}/ingest", files={'files': ('doc_test.pdf', f, 'application/pdf')}, params={"embedding_provider": "gemini"})
        if res.status_code == 200:
            doc_summary = res.json()[0].get('summary')
            print(f"✅ Doc Ingest Success. Summary: {doc_summary[:50]}...")
        else:
            print(f"❌ Doc Ingest Failed: {res.text}")
            return

    print("\n--- Phase 2: RAG Test (Document) ---")
    payload_rag = {
        "question": "What is this document about?",
        "llm_provider": "gemini",
        "data_mode": "document",
        "retrieval_mode": "hierarchical"
    }
    res = requests.post(f"{API_BASE_URL}/chat", json=payload_rag)
    if res.status_code == 200:
        print(f"✅ RAG Answer: {res.json()['answer']}")
    else:
        print(f"❌ RAG Chat Failed: {res.text}")

    print("\n--- Phase 3: Visualizer Test (Tabular) ---")
    payload_viz = {
        "question": "Vẽ biểu đồ hình đường cho Revenue qua các tháng",
        "llm_provider": "gemini",
        "data_mode": "tabular"
    }
    res = requests.post(f"{API_BASE_URL}/chat", json=payload_viz)
    if res.status_code == 200:
        print(f"✅ Viz Success. Answer: {res.json()['answer'][:100]}...")
        if res.json().get('chart_paths'):
            print(f"📈 Charts: {res.json()['chart_paths']}")
        else:
            print("⚠️ No charts generated.")
    else:
        print(f"❌ Viz Chat Failed: {res.text}")

if __name__ == "__main__":
    run_e2e_test()
