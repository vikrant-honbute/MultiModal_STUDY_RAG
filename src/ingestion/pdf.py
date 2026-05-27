from pathlib import Path

from pypdf import PdfReader

from src.config import AppConfig
from src.types import DocumentChunk
from src.utils.chunking import chunk_text


def load_pdf(
    file_path: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[DocumentChunk]:
    cfg = AppConfig() if chunk_size is None or overlap is None else None
    size = chunk_size if chunk_size is not None else cfg.chunk_size
    ov = overlap if overlap is not None else cfg.chunk_overlap

    reader = PdfReader(file_path)
    source_id = Path(file_path).name
    chunks: list[DocumentChunk] = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for idx, chunk in enumerate(chunk_text(page_text, size, ov)):
            chunks.append(
                DocumentChunk(
                    content=chunk,
                    source_id=source_id,
                    source_type="pdf",
                    metadata={"page": page_index, "index": idx},
                )
            )

    return chunks
