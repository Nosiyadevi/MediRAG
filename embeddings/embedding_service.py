"""Embedding backends. Output vectors are L2-normalised float32 (cosine == inner product)."""
import hashlib
import re

import numpy as np


class EmbeddingService:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", query_prefix="", batch_size=32, device=None):
        self.model_name, self.query_prefix = model_name, query_prefix
        self.batch_size, self.device, self._model = batch_size, device, None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts, show_progress=False):
        v = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=show_progress, convert_to_numpy=True)
        return v.astype("float32")

    def encode_query(self, q: str):
        return self.encode_documents([self.query_prefix + q])[0]


class HashingEmbedder:
    """Bag-of-words hashing embedder. TEST/DEMO ONLY (no downloads); not a real semantic model."""
    model_name = "hashing-test-embedder"

    def __init__(self, dim=512):
        self.dim = dim

    def encode_documents(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out

    def encode_query(self, q: str):
        return self.encode_documents([q])[0]


def build_embedder(cfg):
    e = cfg["embedding"]
    if e.get("backend") == "hashing":
        return HashingEmbedder()
    return EmbeddingService(e["model_name"], e.get("query_prefix", ""), e.get("batch_size", 32))
