from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from src.config import AppConfig
from src.types import DocumentChunk


def _extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if "youtube.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[-1]

    if "youtu.be" in host:
        return parsed.path.lstrip("/")

    raise ValueError("Unsupported YouTube URL format.")


def load_youtube(
    url: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[DocumentChunk]:
    cfg = AppConfig() if chunk_size is None or overlap is None else None
    size = chunk_size if chunk_size is not None else cfg.chunk_size
    _ = overlap if overlap is not None else cfg.chunk_overlap

    video_id = _extract_video_id(url)
    segments = YouTubeTranscriptApi.get_transcript(video_id)
    if not segments:
        return []

    chunks: list[DocumentChunk] = []
    current_text = ""
    chunk_start = None
    chunk_end = None

    for segment in segments:
        seg_text = (segment.get("text") or "").replace("\n", " ").strip()
        if not seg_text:
            continue

        if chunk_start is None:
            chunk_start = float(segment.get("start", 0))
        proposed = (current_text + " " + seg_text).strip() if current_text else seg_text

        if len(proposed) <= size or not current_text:
            current_text = proposed
            chunk_end = float(segment.get("start", 0)) + float(segment.get("duration", 0))
            continue

        chunks.append(
            DocumentChunk(
                content=current_text,
                source_id=video_id,
                source_type="youtube",
                metadata={"start": chunk_start, "end": chunk_end},
            )
        )

        current_text = seg_text
        chunk_start = float(segment.get("start", 0))
        chunk_end = float(segment.get("start", 0)) + float(segment.get("duration", 0))

    if current_text:
        chunks.append(
            DocumentChunk(
                content=current_text,
                source_id=video_id,
                source_type="youtube",
                metadata={"start": chunk_start, "end": chunk_end},
            )
        )

    return chunks
