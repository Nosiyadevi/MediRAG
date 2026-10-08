"""Generation / citation / safety metrics.

IMPORTANT: these are transparent lexical PROXIES (token-F1, support ratios), not RAGAS.
RAGAS (LLM-judge) can be run additionally; validate any judge against manual review.
"""
import re

from rag import citation as cit
from .retrieval_metrics import covers


def token_f1(pred: str, gold: str) -> float:
    p, g = cit.tokens(cit.TAG_RE.sub("", pred)), cit.tokens(gold)
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    ov = sum(common.values())
    if ov == 0:
        return 0.0
    pr, rc = ov / len(p), ov / len(g)
    return 2 * pr * rc / (pr + rc)


def expected_numbers_present(answer: str, expected: str) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", expected)
    if not nums:
        return None
    a = {str(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", answer)}
    return sum(str(float(n)) in a for n in nums) / len(nums)


def faithfulness_proxy(answer: str, contexts: list) -> float:
    """Share of answer sentences whose content tokens are >=60% present in the retrieved context."""
    ctx = set(cit.tokens(" ".join(contexts)))
    sents = [s for s in cit.split_sentences(cit.TAG_RE.sub("", answer)) if cit.tokens(s)]
    if not sents:
        return None
    ok = sum(len(set(cit.tokens(s)) & ctx) / len(set(cit.tokens(s))) >= 0.6 for s in sents)
    return ok / len(sents)


def citation_correctness(citations: list, hits_by_tag: dict, golds: list):
    """Share of cited chunks that actually contain the gold evidence."""
    if not citations or not golds:
        return None
    ok = sum(any(covers(hits_by_tag[c["tag"]]["text"], g) for g in golds)
             for c in citations if c["tag"] in hits_by_tag)
    return ok / len(citations)


def citation_completeness(answer: str) -> float:
    sents = [s for s in cit.split_sentences(answer) if cit.tokens(s)]
    return sum(bool(cit.TAG_RE.search(s)) for s in sents) / len(sents) if sents else None
