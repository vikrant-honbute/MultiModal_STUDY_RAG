from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model: str = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    hf_token: str | None = os.getenv("HF_TOKEN")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    use_local_embeddings: bool = _env_bool("USE_LOCAL_EMBEDDINGS", "0")
    allow_local_embeddings_fallback: bool = _env_bool(
        "ALLOW_LOCAL_EMBEDDINGS_FALLBACK", "1"
    )
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
    # Gemini / multi-backend
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    llm_backend: str = os.getenv("LLM_BACKEND", "")
