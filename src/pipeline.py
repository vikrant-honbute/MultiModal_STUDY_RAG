from dataclasses import dataclass

from src.types import AgentResult


@dataclass
class PipelineInput:
    question: str


class RAGPipeline:
    def run(self, inp: PipelineInput) -> AgentResult:
        raise NotImplementedError("Pipeline wiring will be added in the next steps.")
