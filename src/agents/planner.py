from src.types import AgentResult, DocumentChunk


class PlannerAgent:
    def run(self, chunks: list[DocumentChunk]) -> AgentResult:
        raise NotImplementedError("Planner agent not implemented yet.")
