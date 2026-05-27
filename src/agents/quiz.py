from src.llm.generation import LLMClient
from src.types import AgentResult, DocumentChunk
from src.utils.prompting import format_context


class QuizAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, chunks: list[DocumentChunk], difficulty: str, question: str = ""
    ) -> AgentResult:
        if not chunks:
            return AgentResult(content="No relevant context found.")

        context = format_context(chunks)
        focus = question.strip() or "General"
        level = difficulty.strip() or "Intermediate"
        prompt = (
            "You are a quiz generator for students.\n"
            f"Difficulty: {level}\n"
            "Create 8 questions based on the context.\n"
            "Format each question like this:\n"
            "Q1. ...\nA. ...\nB. ...\nC. ...\nD. ...\n"
            "Answer: <letter>\nExplanation: <short>\n\n"
            f"Focus: {focus}\n\n"
            f"Context:\n{context}\n"
        )
        output = self.llm.generate(prompt)
        if not output:
            output = "No output generated."
        return AgentResult(content=output)
