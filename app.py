from pathlib import Path
import tempfile

import streamlit as st

from src.agents.retrieval import RetrievalAgent
from src.config import AppConfig
from src.ingestion.notes import load_notes
from src.ingestion.pdf import load_pdf
from src.ingestion.transcript import load_transcript
from src.ingestion.youtube import load_youtube
from src.vectorstore.faiss_store import FaissStore


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
            errors.append(f"PDF {pdf_file.name}: {exc}")
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
        store.build(chunks)
    except Exception as exc:
        errors.append(f"Embeddings: {exc}")
        return None, errors, len(chunks)

    return store, errors, len(chunks)


def main() -> None:
    st.set_page_config(page_title="Multi-Modal RAG Study Agent", layout="wide")
    cfg = AppConfig()

    st.title("Multi-Modal RAG Study Agent")
    st.caption(
        "Multi-source RAG for study materials. Backend wiring will be added step by step."
    )

    with st.sidebar:
        st.header("Run Mode")
        task = st.selectbox(
            "Task",
            ["Notes", "Quiz", "Flashcards", "Timestamps", "Study Plan"],
        )
        difficulty = st.selectbox(
            "Quiz Difficulty",
            ["Beginner", "Intermediate", "Exam"],
            index=1,
        )
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=5)
        st.text_input("Optional access key", type="password", disabled=True)

    tab_ingest, tab_ask, tab_tools, tab_settings = st.tabs(
        ["Ingest", "Ask", "Study Tools", "Settings"]
    )

    with tab_ingest:
        st.subheader("Add your materials")
        youtube_url = st.text_input("YouTube URL")
        pdf_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        notes_text = st.text_area("Paste notes", height=160)
        transcript_text = st.text_area("Paste transcript", height=160)

        if st.button("Build Index"):
            with st.spinner("Building index..."):
                store, errors, chunk_count = build_index(
                    youtube_url=youtube_url,
                    pdf_files=pdf_files,
                    notes_text=notes_text,
                    transcript_text=transcript_text,
                )

            if errors:
                for error in errors:
                    st.error(error)

            if store is None:
                st.warning("Index not created. Check errors above.")
            else:
                st.session_state["store"] = store
                st.session_state["index_ready"] = True
                st.session_state["chunk_count"] = chunk_count
                st.success(f"Index created with {chunk_count} chunks.")

        st.write(
            "Sources added:",
            bool(youtube_url),
            len(pdf_files or []),
            bool(notes_text),
            bool(transcript_text),
        )

    with tab_ask:
        st.subheader("Ask a question")
        question = st.text_input("Question")
        if st.button("Run Retrieval"):
            store = st.session_state.get("store")
            if not store:
                st.warning("Please build the index first.")
            elif not question:
                st.warning("Enter a question.")
            else:
                agent = RetrievalAgent(store)
                results = agent.run(question, k=top_k)
                if not results:
                    st.info("No results found.")
                for idx, chunk in enumerate(results, start=1):
                    st.markdown(
                        f"**Result {idx}** — {chunk.source_type} / {chunk.source_id}"
                    )
                    st.write(chunk.content)
                    st.json(chunk.metadata)

    with tab_tools:
        st.subheader("Generate study outputs")
        st.write("Selected task:", task)
        if task == "Quiz":
            st.write("Difficulty:", difficulty)
        st.info("Agent wiring will be added next.")

    with tab_settings:
        st.subheader("Configuration")
        st.write("Embedding model:", cfg.embedding_model)
        st.write("LLM model:", cfg.llm_model)
        st.write("HF token set:", bool(cfg.hf_token))
        st.write("Index ready:", bool(st.session_state.get("index_ready")))
        st.write("Chunk count:", int(st.session_state.get("chunk_count", 0)))
        if not cfg.hf_token:
            st.warning(
                "HF_TOKEN is not set. Embeddings may fail for gated models or rate limits."
            )


if __name__ == "__main__":
    main()
