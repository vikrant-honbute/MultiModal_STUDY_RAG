import streamlit as st

from src.config import AppConfig


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
            st.session_state["index_ready"] = True
            st.success("Index created (placeholder).")

        st.write(
            "Sources added:",
            bool(youtube_url),
            len(pdf_files),
            bool(notes_text),
            bool(transcript_text),
        )

    with tab_ask:
        st.subheader("Ask a question")
        question = st.text_input("Question")
        if st.button("Run Pipeline"):
            if not st.session_state.get("index_ready"):
                st.warning("Please build the index first.")
            elif not question:
                st.warning("Enter a question.")
            else:
                st.info("Pipeline not wired yet.")

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


if __name__ == "__main__":
    main()
