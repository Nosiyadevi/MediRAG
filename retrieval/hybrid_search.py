"""Hybrid retrieval: vector + BM25 fused with Reciprocal Rank Fusion (RRF).

Why hybrid: embeddings capture paraphrase/semantics ("low iron anaemia" ~ "iron deficiency"),
BM25 captures exact tokens that embeddings blur: abbreviations (HbA1c), drug names, numbers, test names.
"""


def rrf(rank_lists, k=60):
    scores = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class HybridSearcher:
    def __init__(self, store, bm25=None):
        self.store, self.bm25 = store, bm25

    def search(self, query, top_n=20, hybrid=True, rrf_k=60, doc_filter=None):
        if self.store.index is None:
            return []
        qvec = self.store.embed_query(query)
        v = self.store.search_vec(qvec, top_n, doc_filter)
        vscore = dict(v)
        bscore = {}
        if hybrid and self.bm25 is not None:
            b = self.bm25.search(query, top_n, doc_filter)
            bscore = dict(b)
            fused = rrf([[i for i, _ in v], [i for i, _ in b]], rrf_k)
        else:
            fused = {i: s for i, s in v}
        hits = []
        for idx, fscore in sorted(fused.items(), key=lambda x: -x[1])[:top_n]:
            c = self.store.chunks[idx]
            vs = vscore.get(idx)
            if vs is None:
                vs = self.store.cosine(qvec, idx)
            hits.append({"idx": idx, "text": c["text"], "metadata": c["metadata"], "vector_score": float(vs),
                         "bm25_score": float(bscore[idx]) if idx in bscore else None,
                         "fusion_score": float(fscore)})
        return hits
