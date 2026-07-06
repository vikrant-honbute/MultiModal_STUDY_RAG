"""Multi-Modal RAG Study Agent — Streamlit application."""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from src.config import AppConfig
from src.ingestion.notes import load_notes
from src.ingestion.pdf import load_pdf
from src.ingestion.transcript import load_transcript
from src.ingestion.youtube import load_youtube
from src.pipeline import PipelineInput, RAGPipeline
from src.utils.rendering import render_output
from src.vectorstore.faiss_store import FaissStore, _FAISS_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MultiModal RAG Study Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — modern dark theme with accent colours
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Global page background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: #e2e8f0 !important;
    }

    /* ── Main content cards ── */
    div[data-testid="stVerticalBlock"] > div.stBlock,
    .element-container {
        animation: fadeUp 0.4s ease both;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0);    }
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        font-weight: 600;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.55);
        filter: brightness(1.1);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: rgba(255,255,255,0.08);
        color: #e2e8f0;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.45rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        background: rgba(255,255,255,0.15);
        border-color: rgba(255,255,255,0.3);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        transition: background 0.2s;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(255,255,255,0.1) !important;
    }
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
        color: #cbd5e1 !important;
    }

    /* ── Input fields ── */
    .stTextInput input, .stTextArea textarea, .stSelectbox > div {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        transition: border-color 0.2s;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.25) !important;
    }

    /* ── Alerts ── */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }

    /* ── Metric badge ── */
    .index-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .badge-ready {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-not-ready {
        background: rgba(248, 113, 113, 0.12);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.25);
    }

    /* ── Section headings ── */
    h1 { color: #f1f5f9 !important; letter-spacing: -0.5px; }
    h2 { color: #e2e8f0 !important; }
    h3 { color: #cbd5e1 !important; }
    p, li, label { color: #94a3b8 !important; }
    strong { color: #e2e8f0 !important; }
    code { color: #a78bfa !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        border: 1px dashed rgba(255,255,255,0.15);
        padding: 8px;
    }

    /* ── Slider ── */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    /* ── Success / info / warning ── */
    .stSuccess { background: rgba(52,211,153,0.1) !important; color: #34d399 !important; border-left: 3px solid #34d399 !important; }
    .stInfo    { background: rgba(96,165,250,0.1) !important; color: #60a5fa !important; border-left: 3px solid #60a5fa !important; }
    .stWarning { background: rgba(251,191,36,0.1) !important; color: #fbbf24 !important; border-left: 3px solid #fbbf24 !important; }
    .stError   { background: rgba(248,113,113,0.1) !important; color: #f87171 !important; border-left: 3px solid #f87171 !important; }

    /* ── Table ── */
    .stTable, table { color: #e2e8f0 !important; }
    thead tr th { background: rgba(102,126,234,0.2) !important; color: #c4b5fd !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    youtube_url: str,
    pdf_files: list,
    notes_text: str,
    transcript_text: str,
) -> tuple[FaissStore | None, list[str], int]:
    chunks = []
    errors: list[str] = []

    if youtube_url:
        try:
            chunks.extend(load_youtube(youtube_url))
            st.session_state["last_youtube_url"] = youtube_url
        except Exception as exc:
            errors.append(f"YouTube: {exc}")

    for pdf_file in pdf_files or []:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getbuffer())
                tmp_path = tmp.name
            chunks.extend(load_pdf(tmp_path))
        except Exception as exc:
            errors.append(f"PDF '{pdf_file.name}': {exc}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    if notes_text:
        chunks.extend(load_notes(notes_text))

    if transcript_text:
        chunks.extend(load_transcript(transcript_text))

    if not chunks:
        return None, errors, 0

    store = FaissStore()
    try:
        with st.spinner("Embedding chunks…"):
            store.build(chunks)
    except Exception as exc:
        errors.append(f"Embeddings: {exc}")
        return None, errors, len(chunks)

    return store, errors, len(chunks)


def _index_badge(ready: bool, count: int = 0) -> str:
    if ready:
        return (
            f'<span class="index-badge badge-ready">'
            f'✅ Index ready &nbsp;·&nbsp; {count} chunks</span>'
        )
    return '<span class="index-badge badge-not-ready">⚠️ Index not built</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(cfg: AppConfig) -> tuple[str, str, int, int]:
    with st.sidebar:
        st.markdown("## 📚 Study Agent")
        st.markdown("---")

        # Index status badge
        index_ready: bool = bool(st.session_state.get("index_ready"))
        chunk_count: int = int(st.session_state.get("chunk_count", 0))
        st.markdown(_index_badge(index_ready, chunk_count), unsafe_allow_html=True)
        st.markdown("")

        st.markdown("### ⚙️ Run Settings")
        task = st.selectbox(
            "Task",
            ["Notes", "Quiz", "Flashcards", "Timestamps", "Study Plan"],
            help="Choose what you want the agent to generate.",
        )

        difficulty = "Intermediate"
        if task == "Quiz":
            difficulty = st.selectbox(
                "Quiz Difficulty",
                ["Beginner", "Intermediate", "Exam"],
                index=1,
            )

        top_k = st.slider(
            "Top-k retrieval",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of context chunks passed to the agent.",
        )

        plan_days = 7
        if task == "Study Plan":
            plan_days = st.number_input(
                "Plan length (days)",
                min_value=1,
                max_value=30,
                value=7,
            )

        st.markdown("---")
        st.markdown("### 🤖 Model")
        # Determine active backend for display
        import os as _os
        _gemini_key = cfg.gemini_api_key or _os.getenv("GEMINI_API_KEY", "")
        _backend = cfg.llm_backend or ("gemini" if _gemini_key else "hf")
        if _backend == "gemini":
            st.caption(f"Backend: `Gemini`")
            st.caption(f"Model: `{cfg.gemini_model}`")
            if _gemini_key:
                st.success("GEMINI_API_KEY set ✓", icon="🔑")
            else:
                st.error("GEMINI_API_KEY missing!", icon="❌")
        else:
            st.caption(f"Backend: `HuggingFace`")
            st.caption(f"Model: `{cfg.llm_model}`")
            hf_ok = bool(cfg.hf_token)
            if hf_ok:
                st.success("HF_TOKEN set ✓", icon="🔑")
            else:
                st.warning("HF_TOKEN missing", icon="⚠️")
        st.caption(f"Embed: `{cfg.embedding_model}`")

    return task, difficulty, top_k, int(plan_days)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Ingest
# ─────────────────────────────────────────────────────────────────────────────

def tab_ingest() -> None:
    st.subheader("📥 Add your study materials")
    st.caption("Provide one or more sources, then click **Build Index**.")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        youtube_url = st.text_input(
            "🎬 YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="The transcript will be fetched automatically.",
        )
        notes_text = st.text_area(
            "📝 Paste notes",
            height=160,
            placeholder="Paste your study notes or any text here…",
        )

    with col_right:
        pdf_files = st.file_uploader(
            "📄 Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF lecture slides or textbooks.",
        )
        transcript_text = st.text_area(
            "📜 Paste transcript",
            height=160,
            placeholder="Paste a raw lecture transcript here…",
        )

    st.markdown("")

    sources_ready = any([youtube_url, pdf_files, notes_text, transcript_text])
    build_col, _ = st.columns([1, 3])
    with build_col:
        build_btn = st.button(
            "⚡ Build Index",
            disabled=not sources_ready,
            use_container_width=True,
        )

    if not sources_ready:
        st.info("Add at least one source above to enable indexing.", icon="ℹ️")

    if build_btn:
        with st.spinner("Chunking & indexing your materials…"):
            store, errors, chunk_count = build_index(
                youtube_url=youtube_url,
                pdf_files=pdf_files,
                notes_text=notes_text,
                transcript_text=transcript_text,
            )

        for error in errors:
            st.error(error, icon="❌")

        if store is None:
            st.warning("Index was not created. Check errors above.", icon="⚠️")
        else:
            st.session_state["store"] = store
            st.session_state["index_ready"] = True
            st.session_state["chunk_count"] = chunk_count
            st.success(
                f"Index built successfully — **{chunk_count} chunks** ready.",
                icon="✅",
            )
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Ask
# ─────────────────────────────────────────────────────────────────────────────

def tab_ask(task: str, difficulty: str, top_k: int, plan_days: int) -> None:
    st.subheader("💬 Run the Agent")

    index_ready: bool = bool(st.session_state.get("index_ready"))
    if not index_ready:
        st.warning(
            "Your index is not built yet. Go to the **Ingest** tab first.",
            icon="⚠️",
        )
        return

    question = st.text_input(
        "Optional focus / question",
        placeholder=f"e.g. 'Explain backpropagation' — leave blank for a general {task.lower()}",
    )

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("🚀 Generate", use_container_width=True)

    if run_btn:
        store = st.session_state.get("store")
        if not store:
            st.error("Store not found in session. Please rebuild the index.", icon="❌")
            return

        pipeline = RAGPipeline(store)
        try:
            with st.spinner(f"Running {task} agent…"):
                result = pipeline.run(
                    PipelineInput(
                        task=task,
                        question=question,
                        difficulty=difficulty,
                        top_k=top_k,
                        plan_days=plan_days,
                    )
                )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}", icon="❌")
            return

        # Store last result for potential download
        st.session_state["last_result"] = result
        st.session_state["last_task"] = task

    # Render the last result (persists across reruns)
    result = st.session_state.get("last_result")
    if not result:
        return

    last_task = st.session_state.get("last_task", task)

    st.markdown("---")
    st.subheader(f"📋 {last_task} Output")

    # Determine video_id for timestamp links
    video_id: str | None = None
    yt_url = st.session_state.get("last_youtube_url", "")
    if yt_url:
        try:
            from src.ingestion.youtube import _extract_video_id
            video_id = _extract_video_id(yt_url)
        except Exception:
            pass

    render_output(last_task, result.content, citations=result.citations, video_id=video_id)

    st.markdown("")
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="⬇️ Download output",
            data=result.content,
            file_name=f"{last_task.lower().replace(' ', '_')}_output.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Settings
# ─────────────────────────────────────────────────────────────────────────────

def tab_settings(cfg: AppConfig) -> None:
    st.subheader("⚙️ Configuration")

    import os as _os
    _gemini_key = cfg.gemini_api_key or _os.getenv("GEMINI_API_KEY", "")
    _backend = (cfg.llm_backend or ("gemini" if _gemini_key else "hf")).lower()

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### LLM Backend")
        if _backend == "gemini":
            st.markdown("**Active backend:** `Google Gemini` ✅")
            st.markdown(f"**Gemini model:** `{cfg.gemini_model}`")
            if _gemini_key:
                st.success("GEMINI_API_KEY is set ✓", icon="🔑")
            else:
                st.error(
                    "GEMINI_API_KEY is not set. Add it to your `.env` file.",
                    icon="❌",
                )
            st.info(
                "For HuggingFace Spaces: add `GEMINI_API_KEY` as a **Space Secret** "
                "(Settings → Variables and secrets).",
                icon="🚀",
            )
        else:
            st.markdown("**Active backend:** `HuggingFace Inference API`")
            st.markdown(f"**LLM model:** `{cfg.llm_model}`")
            hf_ok = bool(cfg.hf_token)
            if hf_ok:
                st.success("HF_TOKEN is set ✓", icon="🔑")
            else:
                st.warning(
                    "HF_TOKEN is not set. Add it to your `.env` file.",
                    icon="⚠️",
                )

        st.markdown("**To switch backends**, set in `.env`:")
        st.code("LLM_BACKEND=gemini   # or: hf", language="bash")

    with col2:
        st.markdown("#### Embeddings & Chunking")
        st.markdown(f"**Embedding model:** `{cfg.embedding_model}`")
        st.markdown(f"**Local embedding model:** `{cfg.local_embedding_model}`")
        st.markdown(f"**Use local embeddings:** `{cfg.use_local_embeddings}`")
        st.markdown(f"**Allow local fallback:** `{cfg.allow_local_embeddings_fallback}`")
        st.markdown("")
        st.markdown(f"**Chunk size:** `{cfg.chunk_size}` chars")
        st.markdown(f"**Chunk overlap:** `{cfg.chunk_overlap}` chars")
        st.markdown(f"**Max context chars:** `{cfg.max_context_chars}`")
        st.markdown(f"**LLM max tokens:** `{cfg.llm_max_tokens}`")
        st.markdown(f"**LLM temperature:** `{cfg.llm_temperature}`")

    st.markdown("---")
    st.markdown("#### Runtime Status")
    col3, col4 = st.columns(2, gap="large")

    with col3:
        faiss_status = "✅ Available" if _FAISS_AVAILABLE else "⚠️ Not available (numpy fallback active)"
        st.markdown(f"**FAISS:** {faiss_status}")
        if not _FAISS_AVAILABLE:
            st.info(
                "On Windows, install FAISS manually: `pip install faiss-cpu`. "
                "The numpy fallback works correctly but may be slower on large indexes.",
                icon="ℹ️",
            )

    with col4:
        index_ready = bool(st.session_state.get("index_ready"))
        chunk_count = int(st.session_state.get("chunk_count", 0))
        st.markdown(f"**Index:** {'✅ Ready' if index_ready else '❌ Not built'}")
        if index_ready:
            st.markdown(f"**Chunks:** `{chunk_count}`")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = AppConfig()

    # Hero header
    st.markdown(
        """
        <div style='text-align:center; padding: 2rem 0 1rem;'>
            <h1 style='font-size:2.6rem; font-weight:700; background: linear-gradient(135deg,#667eea,#a78bfa,#ec4899);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.3rem;'>
                📚 Multi-Modal RAG Study Agent
            </h1>
            <p style='color:#94a3b8; font-size:1.05rem; margin:0;'>
                Build study notes, quizzes, flashcards, and study plans from
                YouTube lectures, PDFs, and transcripts — powered by RAG.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    task, difficulty, top_k, plan_days = render_sidebar(cfg)

    tab_ingest_ui, tab_ask_ui, tab_settings_ui = st.tabs(
        ["📥 Ingest", "💬 Ask / Generate", "⚙️ Settings"]
    )

    with tab_ingest_ui:
        tab_ingest()

    with tab_ask_ui:
        tab_ask(task, difficulty, top_k, plan_days)

    with tab_settings_ui:
        tab_settings(cfg)


if __name__ == "__main__":
    main()
