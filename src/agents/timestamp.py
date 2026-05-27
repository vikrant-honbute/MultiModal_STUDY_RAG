from src.llm.generation import LLMClient
from src.types import AgentResult, DocumentChunk
from src.utils.prompting import format_context


class TimestampAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, chunks: list[DocumentChunk], question: str = "") -> AgentResult:
        if not chunks:
            return AgentResult(content="No relevant context found.")

        if not any("start" in (chunk.metadata or {}) for chunk in chunks):
            return AgentResult(content="No timestamped content available.")

        context = format_context(chunks)
        focus = question.strip() or "General"
        prompt = (
            "You map concepts to lecture timestamps.\n"
            "Return a list of key concepts with timestamps from the context.\n"
            "Format: <concept> — <timestamp or range>\n\n"
            f"Focus: {focus}\n\n"
            f"Context:\n{context}\n"
        )
        output = self.llm.generate(prompt)
        if not output:
            output = "No output generated."
        return AgentResult(content=output)
