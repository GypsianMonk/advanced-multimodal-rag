"""
memory/conversation_memory.py
──────────────────────────────
Multi-turn conversation memory.
Tracks dialogue history and injects prior context into follow-up queries
so the RAG system can answer questions like "What about last quarter?"
referencing the prior turn's subject.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class Turn:
    role: str           # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class Conversation:
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    turns: list[Turn] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_turn(self, role: str, content: str, **kwargs) -> Turn:
        turn = Turn(role=role, content=content, **kwargs)
        self.turns.append(turn)
        return turn

    def last_user_query(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.role == "user":
                return turn.content
        return None

    def to_dialogue_string(self, max_turns: int = 6) -> str:
        recent = self.turns[-max_turns:]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "turn_count": len(self.turns),
            "last_query": self.last_user_query(),
        }


class ConversationMemory:
    """
    Manages active conversations and injects dialogue history into queries.
    """

    def __init__(self, max_conversations: int = 500, max_turns_in_context: int = 6):
        self._store: dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self.max_turns_in_context = max_turns_in_context

    def create(self) -> Conversation:
        conv = Conversation()
        self._store[conv.conversation_id] = conv
        if len(self._store) > self.max_conversations:
            oldest = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest]
        logger.debug(f"New conversation: {conv.conversation_id}")
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self._store.get(conversation_id)

    def get_or_create(self, conversation_id: str | None) -> Conversation:
        if conversation_id and conversation_id in self._store:
            return self._store[conversation_id]
        return self.create()

    def build_contextual_query(self, conversation_id: str, new_query: str) -> str:
        """
        Enriches a follow-up query with prior dialogue context.
        E.g. "What about Q2?" → understands subject from previous turn.
        """
        conv = self.get(conversation_id)
        if not conv or len(conv.turns) == 0:
            return new_query

        history = conv.to_dialogue_string(max_turns=self.max_turns_in_context)
        contextual = (
            f"[Conversation history]\n{history}\n\n"
            f"[New question]\n{new_query}"
        )
        return contextual

    def add_user_turn(self, conversation_id: str, query: str) -> None:
        conv = self.get_or_create(conversation_id)
        conv.add_turn("user", query)

    def add_assistant_turn(
        self,
        conversation_id: str,
        answer: str,
        sources: list[str] | None = None,
        confidence: float = 1.0,
    ) -> None:
        conv = self.get_or_create(conversation_id)
        conv.add_turn("assistant", answer, sources=sources or [], confidence=confidence)

    def list_conversations(self) -> list[dict]:
        return [c.summary_dict() for c in self._store.values()]
