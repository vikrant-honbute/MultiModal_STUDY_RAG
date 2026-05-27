from dataclasses import dataclass

from src.agents.citation import CitationAgent
from src.agents.flashcard import FlashcardAgent
from src.agents.notes import NotesAgent
from src.agents.planner import PlannerAgent
from src.agents.quiz import QuizAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.timestamp import TimestampAgent
from src.llm.generation import LLMClient
from src.types import AgentResult
from src.vectorstore.faiss_store import FaissStore


@dataclass
class PipelineInput:
    task: str
    question: str
    difficulty: str = "Intermediate"
    top_k: int = 5
    plan_days: int = 7


class RAGPipeline:
    def __init__(self, store: FaissStore, llm: LLMClient | None = None) -> None:
        self.store = store
        self.llm = llm or LLMClient()
        self.retrieval = RetrievalAgent(store)
        self.notes = NotesAgent(self.llm)
        self.quiz = QuizAgent(self.llm)
        self.flashcards = FlashcardAgent(self.llm)
        self.timestamps = TimestampAgent(self.llm)
        self.planner = PlannerAgent(self.llm)
        self.citations = CitationAgent()

    def run(self, inp: PipelineInput) -> AgentResult:
        query = inp.question.strip() or "overall summary"
        chunks = self.retrieval.run(query, k=inp.top_k)
        if not chunks:
            return AgentResult(content="No relevant context found.", citations=[])

        task = (inp.task or "notes").strip().lower()
        if task == "quiz":
            result = self.quiz.run(chunks, difficulty=inp.difficulty, question=inp.question)
        elif task in ("flashcards", "flashcard"):
            result = self.flashcards.run(chunks, question=inp.question)
        elif task in ("timestamps", "timestamp"):
            result = self.timestamps.run(chunks, question=inp.question)
        elif task in ("study plan", "studyplan", "planner", "plan"):
            result = self.planner.run(
                chunks, question=inp.question, days=inp.plan_days
            )
        else:
            result = self.notes.run(chunks, question=inp.question)

        citation_result = self.citations.run(chunks)
        result.citations = citation_result.citations
        return result
