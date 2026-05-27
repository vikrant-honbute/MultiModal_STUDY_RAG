from src.types import DocumentChunk
from src.vectorstore.faiss_store import FaissStore


class RetrievalAgent:
    def __init__(self, store: FaissStore) -> None:
        self.store = store

    def run(self, query: str, k: int = 5) -> list[DocumentChunk]:
        if not query:
            return []
        return self.store.search(query, k=k)
