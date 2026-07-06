from __future__ import annotations

from huggingface_hub import InferenceClient

from src.config import AppConfig


class EmbeddingModel:
    def __init__(
        self,
        model_id: str | None = None,
        token: str | None = None,
        use_local: bool | None = None,
        allow_fallback: bool | None = None,
        local_model_id: str | None = None,
        batch_size: int = 8,
    ) -> None:
        cfg = None
        if (
            model_id is None
            or token is None
            or use_local is None
            or allow_fallback is None
            or local_model_id is None
        ):
            cfg = AppConfig()

        self.model_id = model_id or cfg.embedding_model
        self.token = token if token is not None else cfg.hf_token
        self.use_local = use_local if use_local is not None else cfg.use_local_embeddings
        self.allow_fallback = (
            allow_fallback
            if allow_fallback is not None
            else cfg.allow_local_embeddings_fallback
        )
        self.local_model_id = local_model_id or cfg.local_embedding_model
        self.batch_size = batch_size
        self._client = InferenceClient(token=self.token) if self.token else InferenceClient()
        self._local_model = None

    def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            result = self._client.feature_extraction(batch, model=self.model_id)
            if result and isinstance(result[0], (int, float)):
                result = [result]
            embeddings.extend([list(map(float, vec)) for vec in result])

        return embeddings

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Local embeddings requested but sentence-transformers is not installed. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        if self._local_model is None:
            self._local_model = SentenceTransformer(self.local_model_id)

        vectors = self._local_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return vectors.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not texts:
            return []

        if self.use_local:
            return self._embed_local(texts)

        try:
            return self._embed_remote(texts)
        except Exception as exc:
            if not self.allow_fallback:
                raise
            try:
                return self._embed_local(texts)
            except Exception as local_exc:
                raise RuntimeError(
                    "Remote embeddings failed and local embeddings are unavailable. "
                    f"Remote error: {exc}. Local error: {local_exc}."
                ) from local_exc
