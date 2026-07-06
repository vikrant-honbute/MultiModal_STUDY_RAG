"""Multi-backend LLM generation client.

Supported backends (set LLM_BACKEND in .env or environment):
  - "gemini"  → Google Gemini API (default when GEMINI_API_KEY is set)
  - "hf"      → HuggingFace Inference API (original backend)

Priority auto-detection order when LLM_BACKEND is not set:
  1. If GEMINI_API_KEY is present → use "gemini"
  2. Otherwise → use "hf"
"""

from __future__ import annotations

import os

from src.config import AppConfig


class LLMClient:
    def __init__(
        self,
        model_id: str | None = None,
        token: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        backend: str | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
    ) -> None:
        cfg = AppConfig()

        self.model_id = model_id or cfg.llm_model
        self.token = token if token is not None else cfg.hf_token
        self.max_new_tokens = max_new_tokens or cfg.llm_max_tokens
        self.temperature = temperature if temperature is not None else cfg.llm_temperature
        self.top_p = top_p if top_p is not None else cfg.llm_top_p

        # ── Resolve backend ──────────────────────────────────────────────────
        self.gemini_api_key = (
            gemini_api_key
            if gemini_api_key is not None
            else os.getenv("GEMINI_API_KEY", "")
        )
        self.gemini_model = gemini_model or os.getenv(
            "GEMINI_MODEL", "gemini-1.5-flash"
        )

        resolved_backend = (
            backend
            or os.getenv("LLM_BACKEND", "")
            or ("gemini" if self.gemini_api_key else "hf")
        )
        self.backend = resolved_backend.strip().lower()

        # ── Initialise clients lazily ────────────────────────────────────────
        self._hf_client = None
        self._gemini_model_obj = None

    # ── HuggingFace ──────────────────────────────────────────────────────────

    def _hf_generate(self, prompt: str) -> str:
        from huggingface_hub import InferenceClient

        if self._hf_client is None:
            self._hf_client = (
                InferenceClient(token=self.token) if self.token else InferenceClient()
            )

        response = self._hf_client.text_generation(
            prompt,
            model=self.model_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return response.strip() if response else ""

    # ── Google Gemini ────────────────────────────────────────────────────────

    def _gemini_generate(self, prompt: str) -> str:
        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. "
                "Run: pip install google-genai"
            ) from exc

        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file or "
                "Space Secrets on HuggingFace."
            )

        client = genai.Client(api_key=self.gemini_api_key)

        response = client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ),
        )
        return response.text.strip() if response and response.text else ""

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        if self.backend == "gemini":
            return self._gemini_generate(prompt)

        # Default: HuggingFace
        return self._hf_generate(prompt)

    @property
    def backend_label(self) -> str:
        """Human-readable label for the active backend."""
        if self.backend == "gemini":
            return f"Gemini ({self.gemini_model})"
        return f"HuggingFace ({self.model_id})"
