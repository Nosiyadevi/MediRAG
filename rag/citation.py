"""Citation engine: evidence tags -> document/page/section/chunk + best supporting sentence,
plus lightweight grounding validation (numbers in the answer must occur in the evidence)."""
import re

TAG_RE = re.compile(r"\[([PK]\d+)\]")
STOP = set("a an the of in on at to for and or is are was were be been it its this that these those with by from as "
           "what which who whom how do does did my me i your you report patient level value result results".split())


def tokens(text: str):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


def split_sentences(text: str):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p and p.strip()]


def best_supporting_sentences(claim: str, text: str, n: int = 1):
    ct = set(tokens(claim))
    scored = []
    for s in split_sentences(text):
        st = set(tokens(s))
        ov = len(ct & st)
        if ov:
            scored.append((s, ov / max(1, len(ct)), -len(st)))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [(s, sc) for s, sc, _ in scored[:n]]


def build_evidence_map(patient_hits, medical_hits):
    m = {f"P{i}": h for i, h in enumerate(patient_hits, start=1)}
    m.update({f"K{i}": h for i, h in enumerate(medical_hits, start=1)})
    return m


def extract_cited_tags(answer: str):
    seen, out = set(), []
    for t in TAG_RE.findall(answer):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def claim_for_tag(answer: str, tag: str, fallback: str) -> str:
    sents = [TAG_RE.sub("", s) for s in split_sentences(answer) if f"[{tag}]" in s]
    return " ".join(sents) or fallback


def citation_entry(tag, hit, claim):
    md = hit["metadata"]
    sup = best_supporting_sentences(claim, hit["text"], 1)
    return {"tag": tag, "document": md.get("document"), "page": md.get("page"), "section": md.get("section"),
            "chunk_id": md.get("chunk_id"), "source_type": md.get("source_type"), "dataset": md.get("dataset"),
            "supporting_sentence": sup[0][0] if sup else None, "evidence_score": hit.get("evidence_score"),
            "url": md.get("url")}


def format_source_line(c) -> str:
    if c["source_type"] == "patient_report":
        return f"{c['document']} — Page {c['page']} — {c['section']} (chunk {c['chunk_id']})"
    return f"{c['dataset']} — {c['document']} (chunk {c['chunk_id']})"


def _norm_num(n: str) -> str:
    try:
        return str(float(n))
    except ValueError:
        return n


def validate_grounding(answer: str, evidence_map: dict):
    """Numbers stated in the answer that appear in none of the evidence chunks."""
    body = TAG_RE.sub("", answer)
    ev = {_norm_num(n) for h in evidence_map.values() for n in re.findall(r"\d+(?:\.\d+)?", h["text"])}
    bad = []
    for n in re.findall(r"\d+(?:\.\d+)?", body):
        if n.isdigit() and int(n) <= 10:
            continue
        if _norm_num(n) not in ev:
            bad.append(n)
    return sorted(set(bad))


def uncited_numeric_sentences(answer: str):
    return [s for s in split_sentences(answer) if re.search(r"\d", s) and not TAG_RE.search(s)]
