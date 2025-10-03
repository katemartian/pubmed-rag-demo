from __future__ import annotations
import os
from typing import Optional

class LLM:
    """
    Minimal adapter: answer(question, context) -> str
    - If OPENAI_API_KEY is set and 'openai' lib available, call OpenAI.
    - Otherwise return a heuristic fallback (first sentence of context).
    """
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._use_openai = False
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI  # type: ignore
                self._client = OpenAI(api_key=api_key)
                self._use_openai = True
            except Exception:
                self._use_openai = False

    def answer(self, question: str, context: str, max_tokens: int = 128) -> str:
        if self._use_openai and self._client:
            # Simple, grounded instruction
            system = (
                "You are a careful assistant. Answer ONLY using the provided context. "
                "If the answer is not present, say 'I cannot find this in the context.'"
            )
            user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer succinctly:"
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass
        # Fallback: take first sentence from context or empty
        sentence = context.split(".")[0].strip()
        return sentence if sentence else ""
