import streamlit as st
import requests
import json
import os
import sys
from pathlib import Path

# Add project root to path to allow importing 'src'
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.core.config import settings
except ImportError:
    st.error(f"Cannot import 'src'. sys.path: {sys.path}")
    st.stop()

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Chat With Data - Hierarchical RAG",
    page_icon="📊",
    layout="wide"
)

# --- Title & Header ---
st.title("📊 Chat With Data")
st.markdown("Hệ thống phân tích dữ liệu đa nguồn (Bảng biểu + Văn bản)")

# --- Sidebar: Configuration & Upload ---
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải lên")
    
    # Provider selection
    llm_provider = st.selectbox(
        "LLM Provider (Graph)",
        options=["gemini", "ollama"],
        index=1 if settings.graph_provider == "ollama" else 0
    )

    memory_provider = st.selectbox(
        "Memory Provider (Mem0)",
        options=["gemini", "ollama"],
        index=1 if settings.memory_provider == "ollama" else 0
    )
    
    st.divider()
    
    # 1. Tabular Upload
    st.subheader("1. Dữ liệu Bảng (CSV/Excel)")
    table_file = st.file_uploader(
        "Upload CSV hoặc Excel",
        type=["csv", "xlsx", "xls"],
        help="Cung cấp dữ liệu để thực hiện các phân tích thống kê và vẽ biểu đồ."
    )
    
    # 2. Document Upload
    st.subheader("2. Tài liệu Văn bản (PDF/DOCX)")
    doc_file = st.file_uploader(
        "Upload PDF hoặc DOCX",
        type=["pdf", "docx"],
        help="Cung cấp ngữ cảnh văn bản để trả lời các câu hỏi về quy trình, chính sách..."
    )

    # 3. Retrieval Mode (for Docs)
    st.subheader("3. Chế độ tìm kiếm (RAG)")
    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        options=["hierarchical", "hybrid"],
        index=0,
        help="Hierarchical: Summary-based context. Hybrid: Vector + Keyword search."
    )
    
    st.divider()
    
    # Ingest Button
    if st.button("🚀 Xử lý dữ liệu", use_container_width=True):
        if not table_file and not doc_file:
            st.error("Vui lòng tải lên ít nhất một file (Bảng biểu hoặc Tài liệu).")
        else:
            with st.spinner("Đang xử lý dữ liệu..."):
                try:
                    # Ingest Table if provided
                    if table_file:
                        files_table = [("files", (table_file.name, table_file.getvalue(), table_file.type))]
                        res_table = requests.post(
                            f"{API_BASE_URL}/ingest",
                            files=files_table,
                            params={"embedding_provider": llm_provider}
                        )
                        
                        if res_table.status_code == 200:
                            st.success(f"✅ Đã xử lý file bảng: {table_file.name}")
                            data = res_table.json()
                            st.session_state["dataframe_head"] = data[0].get("summary", "")
                            st.session_state["dataframe_info"] = data[0].get("info", "")
                            st.session_state["data_mode"] = "tabular"  # Tabular mode takes precedence
                        else:
                            st.error(f"❌ Lỗi xử lý bảng: {res_table.text}")

                    # Ingest Doc if provided
                    if doc_file:
                        files_doc = [("files", (doc_file.name, doc_file.getvalue(), doc_file.type))]
                        res_doc = requests.post(
                            f"{API_BASE_URL}/ingest",
                            files=files_doc,
                            params={"embedding_provider": llm_provider}
                        )
                        if res_doc.status_code == 200:
                            st.success(f"✅ Đã xử lý tài liệu: {doc_file.name}")
                            st.session_state["content_summary"] = res_doc.json()[0].get("summary", "")
                            if "data_mode" not in st.session_state or not table_file:
                                st.session_state["data_mode"] = "document"
                        else:
                            st.error(f"❌ Lỗi xử lý tài liệu: {res_doc.text}")
                    
                    if not table_file and not doc_file:
                        st.session_state["data_mode"] = None

                except Exception as e:
                    st.error(f"🚨 Lỗi kết nối API: {e}")

    if st.button("🗑️ Xóa Memory", type="secondary", use_container_width=True):
        try:
            res = requests.delete(f"{API_BASE_URL}/memory/guest")
            if res.status_code == 200:
                st.toast("Đã xóa bộ nhớ người dùng!")
            else:
                st.error("Không thể xóa bộ nhớ.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

# --- Main Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "charts" in message:
            for chart in message["charts"]:
                # The backend serves charts at /charts/filename
                filename = os.path.basename(chart)
                st.image(f"{API_BASE_URL}/charts/{filename}")

# Chat Input
if prompt := st.chat_input("Hỏi gì đó về dữ liệu của bạn..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Pipeline
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Đang suy nghĩ...")
        
        try:
            payload = {
                "question": prompt,
                "user_id": "guest",
                "llm_provider": llm_provider,
                "memory_provider": memory_provider,
                "content_summary": st.session_state.get("content_summary", ""),
                "dataframe_head": st.session_state.get("dataframe_head", ""),
                "dataframe_info": st.session_state.get("dataframe_info", ""),
                "data_mode": st.session_state.get("data_mode"),
                "retrieval_mode": retrieval_mode
            }
            
            response = requests.post(f"{API_BASE_URL}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "Không có câu trả lời.")
                charts = data.get("chart_paths", [])
                
                message_placeholder.markdown(answer)
                
                if charts:
                    for chart in charts:
                        filename = os.path.basename(chart)
                        st.image(f"{API_BASE_URL}/charts/{filename}")
                
                # Update history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "charts": charts
                })
            else:
                message_placeholder.error(f"❌ Lỗi hệ thống: {response.text}")
        
        except Exception as e:
            message_placeholder.error(f"🚨 Lỗi kết nối: {e}")

# --- Footer ---
st.divider()
st.caption("Powered by LangGraph, LlamaIndex, and Gemini.")
