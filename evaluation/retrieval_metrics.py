"""Retrieval metrics. Relevance is judged against gold evidence SPANS (text), not chunk IDs,
so results stay comparable when chunk size changes."""
import math
import re


def _norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def _toks(t):
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t.lower()))


def covers(chunk_text: str, gold: str, thresh: float = 0.8) -> bool:
    c, g = _norm(chunk_text), _norm(gold)
    if g and g in c:
        return True
    gt = _toks(g)
    return bool(gt) and len(gt & _toks(c)) / len(gt) >= thresh


def relevance_vector(chunk_texts, golds):
    return [int(any(covers(c, g) for g in golds)) for c in chunk_texts]


def precision_at_k(rels, k):
    return sum(rels[:k]) / k if k else 0.0


def recall_at_k(chunk_texts, golds, k):
    if not golds:
        return None
    top = chunk_texts[:k]
    return sum(any(covers(c, g) for c in top) for g in golds) / len(golds)


def mrr(rels):
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels, k, n_gold):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, max(1, n_gold))))
    return dcg / ideal if ideal else 0.0
