"""PDF/TXT/DOCX -> clean text -> section blocks -> chunks + structured extraction."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from extraction.medical_entities import extract_structured
from .chunker import chunk_blocks
from .metadata import extract_patient_id, infer_doc_type
from .pdf_loader import load_pages

SECTION_NAMES = {
    "patient information", "demographics", "diagnoses", "diagnosis", "final diagnosis", "conditions",
    "medications", "discharge medications", "current medications", "laboratory results", "lab results",
    "laboratory findings", "investigations", "hospital course", "discharge plan", "discharge instructions",
    "follow-up", "follow up", "allergies", "procedures", "history", "chief complaint", "vital signs",
    "encounters", "assessment", "plan", "imaging", "discharge summary", "clinical summary",
}


def clean_text(text: str) -> str:
    text = text.replace("\r", "").replace("\x00", "")
    text = "\n".join(l.rstrip() for l in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def is_header(line: str) -> bool:
    s = line.strip().rstrip(":").strip()
    if not s or len(s) > 40:
        return False
    if s.lower() in SECTION_NAMES:
        return True
    return s.isupper() and len(s.split()) <= 4 and not re.search(r"\d", s) and len(s) >= 4


def detect_sections(pages: list) -> list:
    blocks, section = [], "General"
    for p in pages:
        buf = []

        def flush():
            body = "\n".join(buf).strip()
            if body:
                blocks.append({"page": p["page"], "section": section, "text": body})
            buf.clear()

        for line in p["text"].split("\n"):
            if is_header(line):
                flush()
                section = line.strip().rstrip(":").strip().title()
            else:
                buf.append(line)
        flush()
    return blocks


@dataclass
class ProcessedDocument:
    name: str
    path: str
    doc_type: str
    patient_id: str
    pages: list
    blocks: list
    chunks: list
    structured: dict
    warnings: list = field(default_factory=list)


def process_document(path, chunk_tokens=512, overlap_tokens=64,
                     source_type="patient_report", dataset="uploaded") -> ProcessedDocument:
    path = Path(path)
    pages = [{"page": p["page"], "text": clean_text(p["text"])} for p in load_pages(path)]
    warnings = [f"Page {p['page']} has no extractable text" for p in pages if not p["text"]]
    full = "\n".join(p["text"] for p in pages)
    doc_type = infer_doc_type(full, path.name)
    pid = extract_patient_id(full)
    blocks = detect_sections(pages)
    chunks = chunk_blocks(blocks, path.name, chunk_tokens, overlap_tokens, source_type, doc_type, pid, dataset)
    structured = extract_structured(path.name, blocks)
    return ProcessedDocument(path.name, str(path), doc_type, pid, pages, blocks, chunks, structured, warnings)
