from src.types import AgentResult, DocumentChunk


class FlashcardAgent:
    def run(self, chunks: list[DocumentChunk]) -> AgentResult:
        raise NotImplementedError("Flashcard agent not implemented yet.")
