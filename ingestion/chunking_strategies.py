"""
ingestion/chunking_strategies.py
─────────────────────────────────
Advanced chunking strategies beyond simple character splitting:
  - SentenceAwareChunker: respects sentence boundaries
  - SemanticChunker: splits on embedding similarity drops
  - MarkdownChunker: splits on headers
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from loguru import logger


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        ...


class SentenceAwareChunker(BaseChunker):
    """
    Splits text into sentence groups that stay within a token budget.
    Avoids cutting mid-sentence — much better than character splitting.
    """

    def __init__(self, max_tokens: int = 400, overlap_sentences: int = 1):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        pattern = r'(?<=[.!?])\s+'
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        chunks, current, current_tokens = [], [], 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.max_tokens and current:
                chunks.append(" ".join(current))
                # Overlap: keep last N sentences
                current = current[-self.overlap_sentences:]
                current_tokens = sum(self._estimate_tokens(s) for s in current)

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks


class MarkdownChunker(BaseChunker):
    """
    Splits Markdown documents on heading boundaries (## / ###).
    Preserves section context in each chunk.
    """

    def __init__(self, max_tokens: int = 600):
        self.max_tokens = max_tokens

    def chunk(self, text: str) -> list[str]:
        heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
        splits = [(m.start(), m.group()) for m in heading_pattern.finditer(text)]

        if not splits:
            return [text.strip()] if text.strip() else []

        chunks = []
        for i, (start, heading) in enumerate(splits):
            end = splits[i + 1][0] if i + 1 < len(splits) else len(text)
            section = text[start:end].strip()
            if section:
                chunks.append(section)

        return chunks


class SemanticChunker(BaseChunker):
    """
    Splits text where consecutive sentence embeddings diverge sharply.
    Requires a text embedder. Falls back to SentenceAwareChunker if unavailable.
    """

    def __init__(self, embedder=None, threshold: float = 0.75, max_tokens: int = 500):
        self.embedder = embedder
        self.threshold = threshold
        self.fallback = SentenceAwareChunker(max_tokens=max_tokens)

    def chunk(self, text: str) -> list[str]:
        if not self.embedder:
            return self.fallback.chunk(text)

        sentences = self.fallback._split_sentences(text)
        if len(sentences) < 3:
            return [text.strip()]

        embeddings = self.embedder.embed_batch(sentences)

        import numpy as np
        chunks, current = [], [sentences[0]]

        for i in range(1, len(sentences)):
            a = np.array(embeddings[i - 1])
            b = np.array(embeddings[i])
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

            if sim < self.threshold:
                # Semantic break detected — start new chunk
                chunks.append(" ".join(current))
                current = []

            current.append(sentences[i])

        if current:
            chunks.append(" ".join(current))

        logger.debug(f"SemanticChunker: {len(sentences)} sentences → {len(chunks)} chunks")
        return chunks


def get_chunker(strategy: str = "sentence", **kwargs) -> BaseChunker:
    """Factory function for chunker selection via config."""
    strategies = {
        "sentence": SentenceAwareChunker,
        "markdown": MarkdownChunker,
        "semantic": SemanticChunker,
        "recursive": lambda **kw: __import__(
            "utils.chunker", fromlist=["RecursiveChunker"]
        ).RecursiveChunker(**kw),
    }
    cls = strategies.get(strategy)
    if not cls:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Choose from {list(strategies)}")
    return cls(**kwargs)
