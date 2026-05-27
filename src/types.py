from dataclasses import dataclass, field
from typing import Literal

SourceType = Literal["youtube", "pdf", "notes", "transcript"]


@dataclass
class DocumentChunk:
    content: str
    source_id: str
    source_type: SourceType
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks: list[DocumentChunk]


@dataclass
class AgentResult:
    content: str
    citations: list[dict] = field(default_factory=list)
