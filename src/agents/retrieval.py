from src.types import DocumentChunk


class RetrievalAgent:
    def run(self, query: str) -> list[DocumentChunk]:
        raise NotImplementedError("Retrieval agent not implemented yet.")
