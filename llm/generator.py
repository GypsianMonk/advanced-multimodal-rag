"""
llm/generator.py
────────────────
LLM generation layer with structured prompt templating,
context window optimization, and multi-provider support.
"""

from __future__ import annotations

from loguru import logger

SYSTEM_PROMPT = """You are a precise, knowledge-grounded assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain sufficient information, say so clearly.
Do not hallucinate. Cite your sources when possible."""

RAG_TEMPLATE = """{system}

=== CONTEXT ===
{context}
===============

User Question: {query}

Answer:"""


class LLMGenerator:
    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-4-20250514"):
        self.provider = provider
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client:
            return self._client
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif self.provider == "openai":
            import openai
            self._client = openai.OpenAI()
        return self._client

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        prompt = RAG_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            query=query,
        )

        client = self._get_client()

        if self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.provider}")
