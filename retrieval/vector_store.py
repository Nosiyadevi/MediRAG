"""FAISS vector store (exact inner-product index) with chunk metadata kept alongside."""
import json
from pathlib import Path

import faiss
import numpy as np

from ingestion.metadata import chunk_search_text


class VectorStore:
    def __init__(self, embedder):
        self.embedder, self.index, self.chunks = embedder, None, []

    def __len__(self):
        return len(self.chunks)

    def build(self, chunks, show_progress=False):
        self.chunks = list(chunks)
        if not self.chunks:
            self.index = None
            return
        vecs = self.embedder.encode_documents([chunk_search_text(c) for c in self.chunks], show_progress)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def embed_query(self, q):
        return self.embedder.encode_query(q)

    def search_vec(self, qvec, k, doc_filter=None):
        if self.index is None:
            return []
        n = self.index.ntotal
        kk = n if doc_filter else min(k, n)
        scores, ids = self.index.search(np.asarray(qvec, dtype="float32").reshape(1, -1), kk)
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            if doc_filter and self.chunks[i]["metadata"].get("document") != doc_filter:
                continue
            out.append((int(i), float(s)))
            if len(out) >= k:
                break
        return out

    def cosine(self, qvec, idx):
        return float(np.dot(self.index.reconstruct(int(idx)), qvec))

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(d / "index.faiss"))
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(json.dumps({"model": getattr(self.embedder, "model_name", "?"),
                                                 "n_chunks": len(self.chunks)}))

    @classmethod
    def load(cls, directory, embedder):
        d = Path(directory)
        vs = cls(embedder)
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            vs.chunks = [json.loads(l) for l in f if l.strip()]
        if (d / "index.faiss").exists():
            vs.index = faiss.read_index(str(d / "index.faiss"))
        return vs
