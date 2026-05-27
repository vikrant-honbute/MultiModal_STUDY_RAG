from src.types import AgentResult, DocumentChunk


class QuizAgent:
    def run(self, chunks: list[DocumentChunk], difficulty: str) -> AgentResult:
        raise NotImplementedError("Quiz agent not implemented yet.")
