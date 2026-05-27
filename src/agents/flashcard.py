from src.llm.generation import LLMClient
from src.types import AgentResult, DocumentChunk
from src.utils.prompting import format_context


class FlashcardAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, chunks: list[DocumentChunk], question: str = "") -> AgentResult:
        if not chunks:
            return AgentResult(content="No relevant context found.")

        context = format_context(chunks)
        focus = question.strip() or "General"
        prompt = (
            "You are a flashcard creator.\n"
            "Create 10 flashcards from the context.\n"
            "Format exactly as:\n"
            "Front: <short prompt>\nBack: <short answer>\n\n"
            f"Focus: {focus}\n\n"
            f"Context:\n{context}\n"
        )
        output = self.llm.generate(prompt)
        if not output:
            output = "No output generated."
        return AgentResult(content=output)
