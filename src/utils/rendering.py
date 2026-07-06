"""Rich Streamlit renderers for each agent output type.

Each `render_*` function receives the raw LLM content string and a dict of
extra context (e.g. video_id for timestamps) and renders appropriate Streamlit
widgets.  They all return the raw `content` string so callers can still offer a
download.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def render_notes_output(content: str) -> str:
    """Render study notes as formatted markdown."""
    import streamlit as st

    st.markdown(content)
    return content


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

def _parse_quiz_blocks(content: str) -> list[dict]:
    """Parse quiz output into a list of question dicts.

    Expected format per question::

        Q1. <question text>
        A. <option>
        B. <option>
        C. <option>
        D. <option>
        Answer: <letter>
        Explanation: <text>
    """
    blocks: list[dict] = []
    # Split on question headers like "Q1." or "1."
    parts = re.split(r"(?m)^(?:Q?\d+[\.\)])", content)
    # First element before any Q is preamble — skip if empty
    for part in parts:
        part = part.strip()
        if not part:
            continue

        question_match = re.match(r"^(.+?)(?=\nA[\.\)])", part, re.DOTALL)
        question_text = question_match.group(1).strip() if question_match else part.split("\n")[0].strip()

        options: dict[str, str] = {}
        for letter in "ABCD":
            opt_match = re.search(rf"(?m)^{letter}[\.\)]\s*(.+)", part)
            if opt_match:
                options[letter] = opt_match.group(1).strip()

        answer_match = re.search(r"(?i)Answer:\s*([A-Da-d])", part)
        answer = answer_match.group(1).upper() if answer_match else ""

        explanation_match = re.search(r"(?i)Explanation:\s*(.+)", part, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        if question_text and options:
            blocks.append(
                {
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "explanation": explanation,
                }
            )

    return blocks


def render_quiz_output(content: str) -> str:
    """Render quiz as interactive expandable question blocks."""
    import streamlit as st

    blocks = _parse_quiz_blocks(content)

    if not blocks:
        # Fallback: just show raw markdown
        st.markdown(content)
        return content

    st.markdown(f"**{len(blocks)} question(s) generated**")

    for i, block in enumerate(blocks, start=1):
        with st.expander(f"Q{i}. {block['question']}", expanded=(i == 1)):
            for letter, text in block["options"].items():
                is_correct = letter == block["answer"]
                icon = "✅" if is_correct else "⬜"
                st.markdown(f"{icon} **{letter}.** {text}")

            if block["answer"]:
                st.success(f"**Answer: {block['answer']}**")
            if block["explanation"]:
                st.info(f"💡 {block['explanation']}")

    return content


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

def _parse_flashcards(content: str) -> list[tuple[str, str]]:
    """Parse Front:/Back: pairs from content."""
    cards: list[tuple[str, str]] = []
    # Try to find paired Front/Back blocks
    pattern = re.findall(
        r"(?i)Front:\s*(.+?)\s*Back:\s*(.+?)(?=(?:Front:|$))",
        content,
        re.DOTALL,
    )
    for front, back in pattern:
        front = front.strip()
        back = back.strip()
        if front and back:
            cards.append((front, back))
    return cards


def render_flashcard_output(content: str) -> str:
    """Render flashcards as expandable front/back cards."""
    import streamlit as st

    cards = _parse_flashcards(content)

    if not cards:
        st.markdown(content)
        return content

    st.markdown(f"**{len(cards)} flashcard(s) generated**")

    cols_per_row = 2
    for row_start in range(0, len(cards), cols_per_row):
        row_cards = cards[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_cards))
        for col, (front, back) in zip(cols, row_cards):
            with col:
                with st.expander(f"🃏 {front}", expanded=False):
                    st.markdown(f"**Answer:** {back}")

    return content


# ---------------------------------------------------------------------------
# Study Plan
# ---------------------------------------------------------------------------

def _parse_study_plan(content: str) -> list[tuple[str, str]]:
    """Parse a day-by-day study plan into (day_header, body) tuples."""
    # Split on patterns like "Day 1:", "Day 1 —", "**Day 1**"
    parts = re.split(r"(?im)^(?:\*{0,2})(Day\s+\d+[:\—\-]?\s*[^\n]*)\*{0,2}", content)
    days: list[tuple[str, str]] = []
    # parts is: [preamble, header1, body1, header2, body2, ...]
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if header:
            days.append((header, body))
        i += 2
    return days


def render_plan_output(content: str) -> str:
    """Render study plan as a day-by-day timeline."""
    import streamlit as st

    days = _parse_study_plan(content)

    if not days:
        st.markdown(content)
        return content

    st.markdown(f"**{len(days)}-day study plan**")

    for i, (header, body) in enumerate(days, start=1):
        with st.expander(f"📅 {header}", expanded=(i == 1)):
            st.markdown(body if body else "_No details provided._")

    return content


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def render_timestamp_output(content: str, video_id: str | None = None) -> str:
    """Render concept-timestamp pairs as a table with optional YouTube links."""
    import streamlit as st

    # Parse lines of form: <concept> — <timestamp>  or <concept>: <timestamp>
    rows: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match separator — or -  or :
        m = re.match(r"^(.+?)\s*[—\-–:]\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)?)\s*$", line)
        if m:
            concept = m.group(1).strip().lstrip("-•*").strip()
            timestamp = m.group(2).strip()
            rows.append({"Concept": concept, "Timestamp": timestamp})

    if not rows:
        st.markdown(content)
        return content

    st.markdown(f"**{len(rows)} timestamp(s) found**")

    if video_id:
        for row in rows:
            # Convert first timestamp part to seconds for URL
            ts_str = row["Timestamp"].split("-")[0].strip()
            parts_t = ts_str.split(":")
            try:
                if len(parts_t) == 2:
                    secs = int(parts_t[0]) * 60 + int(parts_t[1])
                else:
                    secs = int(parts_t[0]) * 3600 + int(parts_t[1]) * 60 + int(parts_t[2])
                row["Link"] = f"https://youtu.be/{video_id}?t={secs}"
            except ValueError:
                row["Link"] = ""

        for row in rows:
            link_md = f"[▶ {row['Timestamp']}]({row['Link']})" if row.get("Link") else row["Timestamp"]
            st.markdown(f"- **{row['Concept']}** — {link_md}")
    else:
        import pandas as pd
        st.table(pd.DataFrame(rows))

    return content


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def render_output(task: str, content: str, citations: list[dict] | None = None, video_id: str | None = None) -> None:
    """Dispatch to the appropriate renderer based on task name."""
    import streamlit as st

    task_lower = (task or "").strip().lower()

    if task_lower == "quiz":
        render_quiz_output(content)
    elif task_lower in ("flashcards", "flashcard"):
        render_flashcard_output(content)
    elif task_lower in ("study plan", "studyplan", "plan"):
        render_plan_output(content)
    elif task_lower in ("timestamps", "timestamp"):
        render_timestamp_output(content, video_id=video_id)
    else:
        render_notes_output(content)

    if citations:
        with st.expander("📚 Citations", expanded=False):
            st.json(citations)
