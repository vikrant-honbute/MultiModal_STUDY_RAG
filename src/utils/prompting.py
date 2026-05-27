from __future__ import annotations

from src.config import AppConfig
from src.types import DocumentChunk


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    total = int(round(float(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_context(chunks: list[DocumentChunk], max_chars: int | None = None) -> str:
    cfg = AppConfig() if max_chars is None else None
    limit = max_chars if max_chars is not None else cfg.max_context_chars
    if limit is None:
        limit = 0

    parts: list[str] = []
    total = 0
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        meta_bits: list[str] = []
        if "page" in metadata:
            meta_bits.append(f"page {metadata['page']}")
        if "start" in metadata:
            start_ts = format_timestamp(metadata.get("start"))
            end_ts = format_timestamp(metadata.get("end"))
            if end_ts:
                meta_bits.append(f"t={start_ts}-{end_ts}")
            else:
                meta_bits.append(f"t={start_ts}")

        header = f"[{idx}] {chunk.source_type}:{chunk.source_id}"
        if meta_bits:
            header = f"{header} ({', '.join(meta_bits)})"

        body = chunk.content.strip()
        if not body:
            continue
        block = f"{header}\n{body}"

        if limit > 0 and total + len(block) + 2 > limit:
            break

        parts.append(block)
        total += len(block) + 2

    return "\n\n".join(parts)
