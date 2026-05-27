from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AppConfig:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model: str = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    hf_token: str | None = os.getenv("HF_TOKEN")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
