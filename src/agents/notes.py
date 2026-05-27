from src.types import AgentResult, DocumentChunk


class NotesAgent:
    def run(self, chunks: list[DocumentChunk]) -> AgentResult:
        raise NotImplementedError("Notes agent not implemented yet.")
