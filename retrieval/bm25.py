"""BM25 lexical retrieval. Tokeniser keeps numbers, units and hyphenated terms (e.g. 'HbA1c', '10.2', 'g/dL')."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.metadata import chunk_search_text

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


STOP = set("a an the of in on at to for and or is are was were be been it its this that these those with by from as "
           "what which who whom how do does did my me i your you has have had can will would should".split())


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


class BM25Index:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(chunk_search_text(c)) for c in self.chunks]) if self.chunks else None

    def search(self, query, k, doc_filter=None):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        out = []
        for i in np.argsort(-scores):
            if scores[i] <= 0:
                break
            if doc_filter and self.chunks[i]["metadata"].get("document") != doc_filter:
                continue
            out.append((int(i), float(scores[i])))
            if len(out) >= k:
                break
        return out
