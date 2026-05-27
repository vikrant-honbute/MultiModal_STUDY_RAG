from __future__ import annotations

from huggingface_hub import InferenceClient

from src.config import AppConfig


class LLMClient:
    def __init__(
        self,
        model_id: str | None = None,
        token: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> None:
        cfg = (
            AppConfig()
            if model_id is None
            or token is None
            or max_new_tokens is None
            or temperature is None
            or top_p is None
            else None
        )
        self.model_id = model_id or cfg.llm_model
        self.token = token if token is not None else cfg.hf_token
        self.max_new_tokens = max_new_tokens or cfg.llm_max_tokens
        self.temperature = temperature if temperature is not None else cfg.llm_temperature
        self.top_p = top_p if top_p is not None else cfg.llm_top_p
        self._client = InferenceClient(token=self.token) if self.token else InferenceClient()

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return ""

        response = self._client.text_generation(
            prompt,
            model=self.model_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return response.strip() if response else ""
