from src.config import AppConfig
from src.types import DocumentChunk
from src.utils.chunking import chunk_text


def load_notes(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[DocumentChunk]:
    if not text:
        return []

    cfg = AppConfig() if chunk_size is None or overlap is None else None
    size = chunk_size if chunk_size is not None else cfg.chunk_size
    ov = overlap if overlap is not None else cfg.chunk_overlap

    chunks = chunk_text(text, size, ov)
    return [
        DocumentChunk(
            content=chunk,
            source_id="notes",
            source_type="notes",
            metadata={"index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]
