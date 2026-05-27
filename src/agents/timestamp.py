from src.types import AgentResult, DocumentChunk


class TimestampAgent:
    def run(self, chunks: list[DocumentChunk]) -> AgentResult:
        raise NotImplementedError("Timestamp agent not implemented yet.")
