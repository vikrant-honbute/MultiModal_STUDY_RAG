from src.llm.generation import LLMClient
from src.types import AgentResult, DocumentChunk
from src.utils.prompting import format_context


class PlannerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, chunks: list[DocumentChunk], question: str = "", days: int = 7
    ) -> AgentResult:
        if not chunks:
            return AgentResult(content="No relevant context found.")

        context = format_context(chunks)
        focus = question.strip() or "General"
        days = max(1, days)
        prompt = (
            "You are a study planner.\n"
            f"Create a {days}-day study plan based on the context.\n"
            "Include daily topics, estimated time, and review checkpoints.\n"
            "Use a clear day-by-day format.\n\n"
            f"Focus: {focus}\n\n"
            f"Context:\n{context}\n"
        )
        output = self.llm.generate(prompt)
        if not output:
            output = "No output generated."
        return AgentResult(content=output)
