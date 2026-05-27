from __future__ import annotations

from huggingface_hub import InferenceClient

from src.config import AppConfig


class EmbeddingModel:
    def __init__(
        self,
        model_id: str | None = None,
        token: str | None = None,
        batch_size: int = 8,
    ) -> None:
        cfg = AppConfig() if model_id is None or token is None else None
        self.model_id = model_id or cfg.embedding_model
        self.token = token if token is not None else cfg.hf_token
        self.batch_size = batch_size
        self._client = InferenceClient(token=self.token) if self.token else InferenceClient()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self._client.feature_extraction(batch, model=self.model_id)
            if result and isinstance(result[0], (int, float)):
                result = [result]
            embeddings.extend([list(map(float, vec)) for vec in result])

        return embeddings
