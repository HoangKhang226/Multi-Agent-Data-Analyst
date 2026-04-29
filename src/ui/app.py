"""
Chat With Data — Streamlit Frontend
=====================================
Premium UI for the Multi-Agent RAG pipeline.
Features:
  - User profile (user_id, display name)
  - LLM & Embedding provider selection
  - Retrieval mode: hierarchical | hybrid
  - File upload (tabular + document)
  - Memory management (view / clear)
  - Streaming-style chat with chart display
"""

import streamlit as st
import requests
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.core.config import settings
except ImportError:
    st.error("Cannot import 'src'. Please run from project root.")
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000"
Path("storage").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat With Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Header card ── */
.header-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    text-align: center;
}
.header-card h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-card p {
    color: #94a3b8;
    margin: 6px 0 0;
    font-size: 0.95rem;
}

/* ── Section title in sidebar ── */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6366f1 !important;
    margin: 18px 0 6px;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.badge-online  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-offline { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    backdrop-filter: blur(6px);
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.8) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}

/* ── Selectbox & text input ── */
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 16px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 4px; }

/* ── Animation ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.4s ease; }

/* ── Info boxes ── */
.info-box {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #c7d2fe;
    margin-bottom: 8px;
}
.warn-box {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #fde68a;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "messages": [],
    "data_mode": None,
    "content_summary": "",
    "dataframe_head": "",
    "dataframe_info": "",
    "user_id": "guest",
    "api_healthy": None,
    "indexed_files": [],        # list of {filename, collection, data_mode, summary}
    "selected_collection": None, # currently selected collection for chat
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helper: API health ping
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10)
def ping_api():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Logo / brand ──
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px;">
        <span style="font-size:2.4rem;">📊</span>
        <div style="font-size:1.1rem; font-weight:700; color:#a78bfa; margin-top:4px;">Chat With Data</div>
        <div style="font-size:0.72rem; color:#64748b;">Multi-Agent RAG Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API status ──
    api_ok = ping_api()
    badge_cls = "badge-online" if api_ok else "badge-offline"
    badge_txt = "● API Online" if api_ok else "● API Offline"
    st.markdown(
        f'<div style="text-align:center; margin-bottom:12px;">'
        f'<span class="status-badge {badge_cls}">{badge_txt}</span></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── 1. User profile ──
    st.markdown('<div class="sidebar-section">👤 Người dùng</div>', unsafe_allow_html=True)

    user_id = st.text_input(
        "User ID",
        value=st.session_state["user_id"],
        placeholder="e.g. khang_dev",
        help="ID duy nhất xác định người dùng — ảnh hưởng đến long-term memory.",
        label_visibility="collapsed",
    )
    st.session_state["user_id"] = user_id.strip() or "guest"

    st.divider()

    # ── 2. Model configuration ──
    st.markdown('<div class="sidebar-section">🤖 Cấu hình Model</div>', unsafe_allow_html=True)

    llm_provider = st.selectbox(
        "LLM Provider",
        options=["gemini", "ollama"],
        index=0 if settings.graph_provider.lower() == "gemini" else 1,
        help="LLM chạy pipeline agent LangGraph.",
    )

    embedding_provider = st.selectbox(
        "Embedding Provider",
        options=["google", "ollama"],
        index=0 if settings.graph_provider.lower() in ("gemini", "google") else 1,
        help="Model vector hoá văn bản để index & tìm kiếm.",
    )

    memory_provider = st.selectbox(
        "Memory Provider (Mem0)",
        options=["gemini", "ollama"],
        index=0 if settings.memory_provider.lower() == "gemini" else 1,
        help="Provider cho Mem0 long-term memory.",
    )

    st.divider()

    # ── 3. Retrieval strategy ──
    st.markdown('<div class="sidebar-section">🔍 Chiến lược Retrieval</div>', unsafe_allow_html=True)

    retrieval_mode = st.radio(
        "Retrieval Mode",
        options=["hierarchical", "hybrid"],
        index=0,
        horizontal=True,
        help=(
            "**Hierarchical**: Tìm theo cấu trúc phân cấp (parent-child nodes). "
            "Tốt cho tài liệu có cấu trúc rõ ràng.\n\n"
            "**Hybrid**: Kết hợp vector search + BM25 keyword. "
            "Tốt cho câu hỏi đòi hỏi từ khoá chính xác."
        ),
        label_visibility="collapsed",
    )

    # Mode explanation card
    if retrieval_mode == "hierarchical":
        st.markdown(
            '<div class="info-box">🌳 <b>Hierarchical</b>: Truy xuất theo cây phân cấp parent → child nodes. '
            'Phù hợp khi tài liệu có cấu trúc chương/mục rõ ràng.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-box">🔀 <b>Hybrid</b>: Kết hợp dense vector (semantic) + sparse BM25 (keyword). '
            'Tốt cho câu hỏi cần khớp từ khoá chính xác.</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── 4. File upload ──
    st.markdown('<div class="sidebar-section">📁 Tải lên dữ liệu</div>', unsafe_allow_html=True)

    col_t, col_d = st.columns(2)
    with col_t:
        st.caption("📋 Bảng biểu")
    with col_d:
        st.caption("📄 Tài liệu")

    table_file = st.file_uploader(
        "CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Dữ liệu bảng để phân tích thống kê và vẽ biểu đồ.",
        label_visibility="collapsed",
    )
    doc_file = st.file_uploader(
        "PDF / DOCX",
        type=["pdf", "docx"],
        help="Tài liệu văn bản để hỏi đáp.",
        label_visibility="collapsed",
    )

    # Process button
    process_btn = st.button(
        "🚀 Xử lý & Index dữ liệu",
        use_container_width=True,
        type="primary",
        disabled=not api_ok,
    )

    if process_btn:
        if not table_file and not doc_file:
            st.error("⚠️ Vui lòng tải lên ít nhất một file.")
        else:
            with st.spinner("⚙️ Đang xử lý..."):
                try:
                    if table_file:
                        res = requests.post(
                            f"{API_BASE_URL}/ingest",
                            files=[("files", (table_file.name, table_file.getvalue(), table_file.type))],
                            params={"embedding_provider": embedding_provider},
                            timeout=120,
                        )
                        if res.status_code == 200:
                            data = res.json()[0]
                            st.session_state["dataframe_head"] = data.get("summary", "")
                            st.session_state["dataframe_info"] = data.get("info", "")
                            st.session_state["data_mode"] = "tabular"
                            # Track this file
                            coll = data.get("collection", "tabular")
                            existing = [f["collection"] for f in st.session_state["indexed_files"]]
                            if coll not in existing:
                                st.session_state["indexed_files"].append({
                                    "filename": table_file.name,
                                    "collection": coll,
                                    "data_mode": "tabular",
                                    "summary": data.get("summary", ""),
                                    "info": data.get("info", ""),
                                })
                            st.success(f"✅ Bảng biểu: **{table_file.name}**")
                        else:
                            st.error(f"❌ Lỗi bảng: {res.text[:200]}")

                    if doc_file:
                        res = requests.post(
                            f"{API_BASE_URL}/ingest",
                            files=[("files", (doc_file.name, doc_file.getvalue(), doc_file.type))],
                            params={"embedding_provider": embedding_provider},
                            timeout=180,
                        )
                        if res.status_code == 200:
                            data = res.json()[0]
                            st.session_state["content_summary"] = data.get("summary", "")
                            if "data_mode" not in st.session_state or not table_file:
                                st.session_state["data_mode"] = "document"
                            # Track this file
                            coll = data.get("collection", "")
                            existing = [f["collection"] for f in st.session_state["indexed_files"]]
                            if coll and coll not in existing:
                                st.session_state["indexed_files"].append({
                                    "filename": doc_file.name,
                                    "collection": coll,
                                    "data_mode": "document",
                                    "summary": data.get("summary", ""),
                                    "info": "",
                                })
                            st.success(f"✅ Tài liệu: **{doc_file.name}**")
                        else:
                            st.error(f"❌ Lỗi tài liệu: {res.text[:200]}")

                except requests.exceptions.ConnectionError:
                    st.error("🚨 Không kết nối được API. Kiểm tra server đang chạy.")
                except Exception as e:
                    st.error(f"🚨 Lỗi: {e}")

    st.divider()

    # ── 4b. File / Collection selector ──
    st.markdown('<div class="sidebar-section">📂 Tài liệu đang query</div>', unsafe_allow_html=True)

    indexed = st.session_state.get("indexed_files", [])
    if indexed:
        # Build labels for selectbox
        labels = []
        for f in indexed:
            icon = "📋" if f["data_mode"] == "tabular" else "📄"
            labels.append(f"{icon} {f['filename']}")

        # Auto-select last uploaded if nothing selected
        current_coll = st.session_state.get("selected_collection")
        current_idx = 0
        if current_coll:
            colls = [f["collection"] for f in indexed]
            if current_coll in colls:
                current_idx = colls.index(current_coll)

        chosen_label = st.selectbox(
            "Chọn file để query",
            options=labels,
            index=current_idx,
            label_visibility="collapsed",
            help="Mỗi file được index vào collection riêng biệt. Chọn file bạn muốn hỏi.",
        )
        chosen_idx = labels.index(chosen_label)
        chosen_file = indexed[chosen_idx]

        # Update session state from selection
        st.session_state["selected_collection"] = chosen_file["collection"]
        st.session_state["data_mode"] = chosen_file["data_mode"]
        if chosen_file["data_mode"] == "document":
            st.session_state["content_summary"] = chosen_file.get("summary", "")
            st.session_state["dataframe_head"] = ""
            st.session_state["dataframe_info"] = ""
        else:
            st.session_state["content_summary"] = ""
            st.session_state["dataframe_head"] = chosen_file.get("summary", "")
            st.session_state["dataframe_info"] = chosen_file.get("info", "")

        st.markdown(
            f'<div class="info-box">🔑 Collection: <code>{chosen_file["collection"]}</code></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="warn-box">📭 Chưa có file nào được index. Upload file ở trên rồi nhấn Xử lý.</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── 5. Memory management ──
    st.markdown('<div class="sidebar-section">🧠 Quản lý Memory</div>', unsafe_allow_html=True)

    mem_col1, mem_col2 = st.columns(2)

    with mem_col1:
        if st.button("👁 Xem", use_container_width=True, disabled=not api_ok):
            try:
                res = requests.get(
                    f"{API_BASE_URL}/memory/{st.session_state['user_id']}",
                    params={"provider": memory_provider},
                    timeout=10
                )
                if res.status_code == 200:
                    memories = res.json().get("memories", [])
                    st.session_state["_memory_preview"] = memories
                else:
                    st.session_state["_memory_preview"] = []
            except Exception as e:
                st.error(f"Lỗi: {e}")

    with mem_col2:
        if st.button("🗑 Xóa", use_container_width=True, type="secondary", disabled=not api_ok):
            try:
                res = requests.delete(
                    f"{API_BASE_URL}/memory/{st.session_state['user_id']}",
                    params={"provider": memory_provider},
                    timeout=10
                )
                if res.status_code == 200:
                    st.toast("✅ Đã xóa toàn bộ memory!", icon="🗑")
                    st.session_state.pop("_memory_preview", None)
                else:
                    st.error("Không thể xóa memory.")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # Memory preview
    if "_memory_preview" in st.session_state:
        mems = st.session_state["_memory_preview"]
        if mems:
            with st.expander(f"📋 {len(mems)} memories", expanded=True):
                for i, m in enumerate(mems, 1):
                    mem_text = m.get("memory", str(m))
                    st.markdown(f"**{i}.** {mem_text}")
        else:
            st.markdown(
                '<div class="info-box">💭 Chưa có memory nào cho user này.</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── 6. Vector DB reset ──
    st.markdown('<div class="sidebar-section">⚠️ Quản trị</div>', unsafe_allow_html=True)

    if st.button(
        "🔄 Reset Vector DB",
        use_container_width=True,
        type="secondary",
        disabled=not api_ok,
        help="Xóa toàn bộ dữ liệu đã index trong vector database.",
    ):
        try:
            res = requests.delete(
                f"{API_BASE_URL}/reset",
                params={"embedding_provider": embedding_provider},
                timeout=30,
            )
            if res.status_code == 200:
                st.session_state["data_mode"] = None
                st.session_state["content_summary"] = ""
                st.session_state["dataframe_head"] = ""
                st.toast("✅ Vector DB đã được xóa sạch!", icon="🔄")
            else:
                st.error(f"Lỗi reset: {res.text[:200]}")
        except Exception as e:
            st.error(f"Lỗi: {e}")

    # Clear chat
    if st.button("💬 Xóa lịch sử chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    # ── Footer ──
    st.markdown("""
    <div style="text-align:center; padding:16px 0 0; color:#475569; font-size:0.72rem; line-height:1.6;">
        Powered by<br>
        <span style="color:#6366f1;">LangGraph</span> ·
        <span style="color:#0ea5e9;">LlamaIndex</span> ·
        <span style="color:#10b981;">Qdrant</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------

# ── Header card ──
st.markdown("""
<div class="header-card fade-in">
    <h1>📊 Chat With Data</h1>
    <p>Hệ thống phân tích dữ liệu đa nguồn — Hierarchical & Hybrid RAG với Long-Term Memory</p>
</div>
""", unsafe_allow_html=True)

# ── Active config info bar ──
data_mode_display = {
    "tabular": "📋 Tabular (CSV/Excel)",
    "document": "📄 Document (PDF/DOCX)",
    None: "⚠️ Chưa có dữ liệu — hãy upload file ở sidebar",
}

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("👤 User", st.session_state["user_id"])
m2.metric("🤖 LLM", llm_provider.title())
m3.metric("🔢 Embedding", embedding_provider.title())
m4.metric("🔍 Retrieval", retrieval_mode.title())
m5.metric("📁 Data Mode", st.session_state["data_mode"] or "—")

if st.session_state["data_mode"] is None and api_ok:
    st.markdown(
        '<div class="warn-box">⚠️ Chưa có dữ liệu được index. '
        'Hãy upload file CSV/Excel hoặc PDF/DOCX ở sidebar rồi nhấn <b>Xử lý & Index dữ liệu</b>.</div>',
        unsafe_allow_html=True
    )

st.divider()

# ── Chat history ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display charts if any
        if message.get("charts"):
            for chart_path in message["charts"]:
                filename = os.path.basename(chart_path)
                chart_url = f"{API_BASE_URL}/charts/{filename}"
                try:
                    st.image(chart_url, use_container_width=True)
                except Exception:
                    st.markdown(f"[📊 Xem biểu đồ]({chart_url})")

# ── Chat input ──
if not api_ok:
    st.markdown(
        '<div class="warn-box">🚨 API đang offline. '
        'Chạy <code>python -m src.api.main</code> để khởi động server.</div>',
        unsafe_allow_html=True
    )

prompt = st.chat_input(
    "Hỏi gì đó về dữ liệu của bạn...",
    disabled=not api_ok,
)

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            "🔍 *Đang phân tích câu hỏi và tìm kiếm thông tin...*"
        )

        payload = {
            "question": prompt,
            "user_id": st.session_state["user_id"],
            "llm_provider": llm_provider,
            "memory_provider": memory_provider,
            "embedding_provider": embedding_provider,
            "content_summary": st.session_state.get("content_summary", ""),
            "dataframe_head": st.session_state.get("dataframe_head", ""),
            "dataframe_info": st.session_state.get("dataframe_info", ""),
            "data_mode": st.session_state.get("data_mode"),
            "retrieval_mode": retrieval_mode,
            "collection_name": st.session_state.get("selected_collection"),
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json=payload,
                timeout=300,
            )

            thinking_placeholder.empty()

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer") or "*(Không có câu trả lời.)*"
                charts = data.get("chart_paths", [])
                sub_tasks = data.get("sub_tasks", [])
                meta = data.get("meta", {})
                is_ambiguous = data.get("is_ambiguous", False)
                rejection = data.get("rejection_reason")

                # Show ambiguity warning
                if is_ambiguous and rejection:
                    st.warning(f"⚠️ Câu hỏi chưa rõ ràng: {rejection}")

                # Show sub-tasks if any
                if sub_tasks:
                    with st.expander("🔧 Sub-tasks được thực thi", expanded=False):
                        for i, task in enumerate(sub_tasks, 1):
                            st.markdown(f"**{i}.** {task}")

                # Main answer
                st.markdown(answer)

                # Charts
                if charts:
                    st.markdown("---")
                    st.markdown("**📊 Biểu đồ kết quả:**")
                    for chart_path in charts:
                        filename = os.path.basename(chart_path)
                        chart_url = f"{API_BASE_URL}/charts/{filename}"
                        try:
                            st.image(chart_url, use_container_width=True)
                        except Exception:
                            st.markdown(f"[📊 Tải biểu đồ]({chart_url})")

                # Meta info
                with st.expander("ℹ️ Chi tiết thực thi", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("LLM", meta.get("llm", llm_provider))
                    c2.metric("Embedding", meta.get("embedding", embedding_provider))
                    c3.metric("User", meta.get("user_id", st.session_state["user_id"]))

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "charts": charts,
                })

            else:
                error_msg = f"❌ Lỗi API ({response.status_code}): {response.text[:400]}"
                thinking_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except requests.exceptions.Timeout:
            msg = "⏱️ Timeout — Server đang xử lý quá lâu. Thử lại với câu hỏi ngắn hơn."
            thinking_placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        except requests.exceptions.ConnectionError:
            msg = "🚨 Không kết nối được API. Kiểm tra server tại `http://localhost:8000`."
            thinking_placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        except Exception as e:
            msg = f"🚨 Lỗi không xác định: {e}"
            thinking_placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
