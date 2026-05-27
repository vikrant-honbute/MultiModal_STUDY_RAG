from __future__ import annotations

from typing import Optional

import numpy as np

from src.types import DocumentChunk
from src.vectorstore.embeddings import EmbeddingModel

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - handled by fallback
    faiss = None
    _FAISS_AVAILABLE = False


class FaissStore:
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        normalize: bool = True,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.normalize = normalize
        self._index = None
        self._vectors: np.ndarray | None = None
        self._chunks: list[DocumentChunk] = []
        self._use_faiss = _FAISS_AVAILABLE

    def build(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._index = None
            self._vectors = None
            return

        embeddings = self.embedding_model.embed([c.content for c in self._chunks])
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.size == 0:
            self._index = None
            self._vectors = None
            return

        if self.normalize:
            if self._use_faiss:
                faiss.normalize_L2(vectors)
            else:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms

        if self._use_faiss:
            dim = vectors.shape[1]
            self._index = faiss.IndexFlatIP(dim) if self.normalize else faiss.IndexFlatL2(dim)
            self._index.add(vectors)
            self._vectors = None
        else:
            self._index = None
            self._vectors = vectors

    def search(self, query: str, k: int = 5) -> list[DocumentChunk]:
        if not query or k <= 0 or not self._chunks:
            return []

        if self._index is None and self._vectors is None:
            return []

        query_vecs = self.embedding_model.embed([query])
        if not query_vecs:
            return []

        query_vec = np.asarray(query_vecs, dtype="float32")
        if self.normalize:
            if self._use_faiss:
                faiss.normalize_L2(query_vec)
            else:
                norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                query_vec = query_vec / norms

        top_k = min(k, len(self._chunks))
        results: list[DocumentChunk] = []

        if self._use_faiss and self._index is not None:
            distances, indices = self._index.search(query_vec, top_k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                chunk = self._chunks[int(idx)]
                metadata = dict(chunk.metadata)
                metadata["score"] = float(score)
                results.append(
                    DocumentChunk(
                        content=chunk.content,
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        metadata=metadata,
                    )
                )
        elif self._vectors is not None:
            scores = np.dot(self._vectors, query_vec[0])
            top_indices = np.argsort(-scores)[:top_k]
            for idx in top_indices:
                chunk = self._chunks[int(idx)]
                metadata = dict(chunk.metadata)
                metadata["score"] = float(scores[idx])
                results.append(
                    DocumentChunk(
                        content=chunk.content,
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        metadata=metadata,
                    )
                )

        return results
