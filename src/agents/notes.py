from src.llm.generation import LLMClient
from src.types import AgentResult, DocumentChunk
from src.utils.prompting import format_context


class NotesAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, chunks: list[DocumentChunk], question: str = "") -> AgentResult:
        if not chunks:
            return AgentResult(content="No relevant context found.")

        context = format_context(chunks)
        focus = question.strip() or "General"
        prompt = (
            "You are a study assistant. Create concise study notes from the context.\n"
            "Requirements:\n"
            "- Use short headings\n"
            "- Bullet points\n"
            "- Include definitions, formulas, and key terms\n"
            "- Keep it concise and factual\n\n"
            f"Focus: {focus}\n\n"
            f"Context:\n{context}\n"
        )
        output = self.llm.generate(prompt)
        if not output:
            output = "No output generated."
        return AgentResult(content=output)
