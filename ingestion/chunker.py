"""Structure-preserving chunker.

Lines (e.g. lab-table rows) are atomic units and are never split unless a single
line exceeds the chunk size. Token counts are approximated by words / 0.75.
"""
from .metadata import build_metadata, make_chunk_id


def tokens_to_words(tokens: int) -> int:
    return max(1, int(tokens * 0.75))


def _units(text: str, max_words: int):
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        w = line.split()
        if len(w) <= max_words:
            yield line
        else:
            for i in range(0, len(w), max_words):
                yield " ".join(w[i:i + max_words])


def chunk_text(text: str, chunk_tokens: int = 512, overlap_tokens: int = 64) -> list:
    max_w = tokens_to_words(chunk_tokens)
    ov_w = min(tokens_to_words(overlap_tokens), max_w // 2)
    chunks, cur, cur_len = [], [], 0
    for u in _units(text, max_w):
        n = len(u.split())
        if cur and cur_len + n > max_w:
            chunks.append("\n".join(cur))
            carry, total = [], 0
            for prev in reversed(cur):  # overlap = trailing units
                pn = len(prev.split())
                if total + pn > ov_w:
                    break
                carry.insert(0, prev)
                total += pn
            cur, cur_len = carry, total
        cur.append(u)
        cur_len += n
    if cur:
        tail = "\n".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def chunk_blocks(blocks: list, document: str, chunk_tokens: int, overlap_tokens: int,
                 source_type="patient_report", doc_type="medical_record",
                 patient_id=None, dataset="uploaded") -> list:
    """blocks: [{"page","section","text"}] -> chunks [{"text","metadata"}]."""
    counters, out = {}, []
    for b in blocks:
        for piece in chunk_text(b["text"], chunk_tokens, overlap_tokens):
            n = counters.get(b["page"], 0) + 1
            counters[b["page"]] = n
            out.append({"text": piece, "metadata": build_metadata(
                document=document, page=b["page"], section=b["section"],
                chunk_id=make_chunk_id(document, b["page"], n), source_type=source_type,
                doc_type=doc_type, patient_id=patient_id, dataset=dataset)})
    return out
