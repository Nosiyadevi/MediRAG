"""Metadata helpers. Every chunk keeps the metadata needed for citations."""
import re
from pathlib import Path


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def make_chunk_id(doc_name: str, page: int, idx: int) -> str:
    return f"{slugify(Path(doc_name).stem)}_p{page}_c{idx}"


PATIENT_ID_RE = re.compile(r"(?:patient\s*id|mrn|patient\s*number)\s*[:#]\s*([A-Za-z0-9\-_]+)", re.I)


def extract_patient_id(text: str):
    m = PATIENT_ID_RE.search(text)
    return m.group(1) if m else None


def infer_doc_type(text: str, filename: str = "") -> str:
    t = (text[:1500] + " " + filename).lower()
    if "discharge summary" in t:
        return "discharge_summary"
    if "laboratory report" in t or "lab report" in t or "cbc" in t:
        return "lab_report"
    if "clinical report" in t:
        return "clinical_report"
    return "medical_record"


def build_metadata(*, document, page, section, chunk_id, source_type,
                   doc_type="unknown", patient_id=None, dataset="uploaded", extra=None) -> dict:
    md = {
        "document": document, "page": page, "section": section, "chunk_id": chunk_id,
        "source_type": source_type, "doc_type": doc_type, "patient_id": patient_id,
        "dataset": dataset,
    }
    if extra:
        md.update(extra)
    return md


def chunk_search_text(chunk: dict) -> str:
    """Text used for embedding / BM25: section title helps retrieval for patient reports."""
    md = chunk["metadata"]
    if md.get("source_type") == "patient_report" and md.get("section"):
        return f"{md['section']}: {chunk['text']}"
    return chunk["text"]
