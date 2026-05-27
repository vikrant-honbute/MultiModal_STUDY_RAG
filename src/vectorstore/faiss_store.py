from src.types import DocumentChunk


class FaissStore:
    def build(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError("FAISS index not implemented yet.")

    def search(self, query: str, k: int = 5) -> list[DocumentChunk]:
        raise NotImplementedError("FAISS search not implemented yet.")
