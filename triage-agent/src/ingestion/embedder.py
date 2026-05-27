from typing import List, Dict, Sequence
import math

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback when model packages are unavailable
    SentenceTransformer = None


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fallback_dim: int = 384):
        self.model_name = model_name
        self.fallback_dim = fallback_dim
        self.model = SentenceTransformer(model_name) if SentenceTransformer is not None else None

    def _fallback_vector(self, text: str) -> List[float]:
        vector = [0.0] * self.fallback_dim
        for token in text.lower().split():
            index = abs(hash(token)) % self.fallback_dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _encode(self, texts: Sequence[str]):
        if self.model is None:
            return [self._fallback_vector(text) for text in texts]
        try:
            return self.model.encode(list(texts), batch_size=64, show_progress_bar=False)
        except Exception:
            return [self._fallback_vector(text) for text in texts]

    def embed_query(self, text: str):
        return self._encode([text])[0]

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 64):
        texts = [c["text"] for c in chunks]
        if self.model is None:
            return [self._fallback_vector(text) for text in texts]
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            try:
                import numpy as np

                return np.array(embeddings, dtype=np.float32)
            except Exception:
                return [list(map(float, row)) for row in embeddings]
        except Exception:
            return [self._fallback_vector(text) for text in texts]
