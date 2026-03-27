"""
embeddings/image_embedder.py
────────────────────────────
CLIP-based image and image-text embedding.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger


class ImageEmbedder:
    """Uses OpenAI CLIP for image and cross-modal embeddings."""

    def __init__(self, model_name: str = "ViT-B/32"):
        import clip
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.dim = 512  # ViT-B/32 output dim
        logger.success("Image embedder (CLIP) ready")

    def embed_image(self, image_path: str) -> list[float]:
        import clip
        import torch
        from PIL import Image

        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model.encode_image(image)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.squeeze().cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        import clip
        import torch

        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            vec = self.model.encode_text(tokens)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.squeeze().cpu().tolist()

    def image_text_similarity(self, image_path: str, text: str) -> float:
        import numpy as np
        img_vec = self.embed_image(image_path)
        txt_vec = self.embed_text(text)
        return float(np.dot(img_vec, txt_vec))
