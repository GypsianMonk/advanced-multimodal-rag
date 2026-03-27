"""
ingestion/ingest.py
───────────────────
Document ingestion pipeline. Loads text, images, and tables
from multiple file formats, chunks them, and persists embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterator

from loguru import logger

from embeddings.text_embedder import TextEmbedder
from embeddings.image_embedder import ImageEmbedder
from utils.chunker import RecursiveChunker
from utils.helpers import load_config


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_pdf(path: Path) -> Iterator[dict]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            yield {"content": text, "source": str(path), "page": i + 1, "type": "text"}


def load_docx(path: Path) -> Iterator[dict]:
    import docx
    doc = docx.Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    yield {"content": full_text, "source": str(path), "page": 1, "type": "text"}


def load_image(path: Path) -> Iterator[dict]:
    yield {"content": str(path), "source": str(path), "page": 1, "type": "image"}


def load_csv(path: Path) -> Iterator[dict]:
    import pandas as pd
    df = pd.read_csv(path)
    # Convert table to text representation for embedding
    text = df.to_string(index=False)
    yield {"content": text, "source": str(path), "page": 1, "type": "table"}


LOADER_MAP = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".doc": load_docx,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".csv": load_csv,
}


# ─── Ingestor ────────────────────────────────────────────────────────────────

class Ingestor:
    def __init__(self, config: dict):
        self.cfg = config
        self.chunker = RecursiveChunker(
            chunk_size=config.get("chunk_size", 512),
            overlap=config.get("chunk_overlap", 64),
        )
        self.text_embedder = TextEmbedder(config["models"]["text_embedding"])
        self.image_embedder = ImageEmbedder(config["models"]["image_embedding"])

    def _doc_id(self, content: str, source: str) -> str:
        return hashlib.md5(f"{source}::{content[:64]}".encode()).hexdigest()

    def ingest_file(self, path: Path) -> list[dict]:
        suffix = path.suffix.lower()
        loader = LOADER_MAP.get(suffix)
        if not loader:
            logger.warning(f"No loader for {suffix} — skipping {path.name}")
            return []

        documents = []
        for raw in loader(path):
            if raw["type"] == "image":
                emb = self.image_embedder.embed_image(raw["content"])
                doc = {**raw, "embedding": emb, "id": self._doc_id(raw["content"], raw["source"])}
            else:
                chunks = self.chunker.chunk(raw["content"])
                for chunk in chunks:
                    emb = self.text_embedder.embed(chunk)
                    doc = {
                        **raw,
                        "content": chunk,
                        "embedding": emb,
                        "id": self._doc_id(chunk, raw["source"]),
                    }
                    documents.append(doc)
                continue
            documents.append(doc)

        logger.info(f"Ingested {len(documents)} chunks from {path.name}")
        return documents

    def ingest_directory(self, source_dir: str) -> list[dict]:
        all_docs = []
        for path in Path(source_dir).rglob("*"):
            if path.is_file() and path.suffix.lower() in LOADER_MAP:
                all_docs.extend(self.ingest_file(path))
        logger.success(f"Total ingested: {len(all_docs)} documents")
        return all_docs


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--source", required=True, help="Source directory or file")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ingestor = Ingestor(config)
    ingestor.ingest_directory(args.source)


if __name__ == "__main__":
    main()
