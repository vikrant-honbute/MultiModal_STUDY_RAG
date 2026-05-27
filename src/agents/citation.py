from src.types import AgentResult, DocumentChunk


class CitationAgent:
    def run(self, chunks: list[DocumentChunk]) -> AgentResult:
        raise NotImplementedError("Citation agent not implemented yet.")
