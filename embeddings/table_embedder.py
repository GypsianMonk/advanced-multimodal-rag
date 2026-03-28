"""
embeddings/table_embedder.py
─────────────────────────────
Converts structured table/dataframe data into text-based embeddings.
Supports multiple serialization strategies for optimal retrieval.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from embeddings.text_embedder import TextEmbedder


class TableSerializer:
    """Converts a DataFrame into a text string optimised for embedding."""

    @staticmethod
    def to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
        return df.head(max_rows).to_markdown(index=False)

    @staticmethod
    def to_summary(df: pd.DataFrame) -> str:
        lines = [
            f"Table with {len(df)} rows and {len(df.columns)} columns.",
            f"Columns: {', '.join(df.columns.tolist())}.",
        ]
        for col in df.select_dtypes(include="number").columns:
            lines.append(
                f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
            )
        return " ".join(lines)

    @staticmethod
    def to_row_sentences(df: pd.DataFrame, max_rows: int = 10) -> str:
        sentences = []
        for _, row in df.head(max_rows).iterrows():
            parts = [f"{col} is {val}" for col, val in row.items()]
            sentences.append("; ".join(parts) + ".")
        return " ".join(sentences)


class TableEmbedder:
    """Embeds tables using text serialisation + Sentence Transformers."""

    STRATEGIES = ("markdown", "summary", "rows")

    def __init__(self, text_embedder: TextEmbedder, strategy: str = "summary"):
        assert strategy in self.STRATEGIES, f"Strategy must be one of {self.STRATEGIES}"
        self.text_embedder = text_embedder
        self.strategy = strategy

    def embed_dataframe(self, df: pd.DataFrame) -> list[float]:
        if self.strategy == "markdown":
            text = TableSerializer.to_markdown(df)
        elif self.strategy == "summary":
            text = TableSerializer.to_summary(df)
        else:
            text = TableSerializer.to_row_sentences(df)

        logger.debug(f"Embedding table via '{self.strategy}' strategy ({len(text)} chars)")
        return self.text_embedder.embed(text)

    def embed_csv(self, csv_path: str) -> tuple[list[float], str]:
        df = pd.read_csv(csv_path)
        text = TableSerializer.to_summary(df)
        embedding = self.text_embedder.embed(text)
        return embedding, text
