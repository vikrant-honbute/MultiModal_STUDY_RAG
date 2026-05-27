from src.types import AgentResult, DocumentChunk
from src.utils.prompting import format_timestamp


class CitationAgent:
    def run(self, chunks: list[DocumentChunk]) -> AgentResult:
        if not chunks:
            return AgentResult(content="No citations available.", citations=[])

        citations: list[dict] = []
        seen: set[tuple] = set()
        lines: list[str] = []

        for chunk in chunks:
            metadata = chunk.metadata or {}
            entry: dict = {
                "source_type": chunk.source_type,
                "source_id": chunk.source_id,
            }
            for key in ("page", "start", "end", "score", "index"):
                if key in metadata:
                    entry[key] = metadata[key]

            key_tuple = tuple(sorted(entry.items()))
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
            citations.append(entry)

            label = f"{entry['source_type']}:{entry['source_id']}"
            if "page" in entry:
                label = f"{label} page {entry['page']}"
            if "start" in entry:
                start_ts = format_timestamp(entry.get("start"))
                end_ts = format_timestamp(entry.get("end"))
                label = f"{label} t={start_ts}"
                if end_ts:
                    label = f"{label}-{end_ts}"
            lines.append(f"- {label}")

        content = "\n".join(lines) if lines else "No citations available."
        return AgentResult(content=content, citations=citations)
