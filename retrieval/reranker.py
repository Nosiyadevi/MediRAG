"""Cross-encoder reranking. 'evidence_score' in [0,1] is a relevance score, NOT a calibrated confidence."""
import math

from ingestion.metadata import chunk_search_text


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name, self._model = model_name, None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            ident = torch.nn.Identity()
            try:  # sentence-transformers >= 4
                self._model = CrossEncoder(self.model_name, max_length=512, activation_fn=ident)
            except TypeError:  # sentence-transformers 2.x / 3.x
                self._model = CrossEncoder(self.model_name, max_length=512, default_activation_function=ident)
        return self._model

    def rerank(self, query, hits, top_k=5):
        if not hits:
            return []
        pairs = [(query, chunk_search_text(h)) for h in hits]
        raw = self.model.predict(pairs)
        out = []
        for h, r in zip(hits, raw):
            h = dict(h)
            h["rerank_score"] = float(r)
            h["evidence_score"] = _sigmoid(float(r))
            out.append(h)
        out.sort(key=lambda h: -h["rerank_score"])
        return out[:top_k]


def passthrough(hits, top_k=5):
    out = []
    for h in hits[:top_k]:
        h = dict(h)
        h["rerank_score"] = None
        h["evidence_score"] = max(0.0, min(1.0, h["vector_score"]))
        out.append(h)
    return out
