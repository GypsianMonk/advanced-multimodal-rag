"""utils/chunker.py — Recursive text chunker with overlap."""

from __future__ import annotations


class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.overlap
        return chunks
