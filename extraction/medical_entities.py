"""Structured information extraction from section blocks (rule-based)."""
import re

from .dates import find_dates, first_date, iso
from .lab_extractor import extract_labs, REPORT_DATE_RE

SECTION_MAP = {
    "patient_information": {"patient information", "demographics"},
    "diagnoses": {"diagnoses", "diagnosis", "final diagnosis", "conditions", "assessment"},
    "medications": {"medications", "discharge medications", "current medications"},
    "procedures": {"procedures"},
    "encounters": {"encounters", "hospital course"},
    "follow_up": {"follow-up", "follow up", "discharge plan", "discharge instructions"},
}
KV_RE = re.compile(r"^([A-Za-z][A-Za-z ]{1,30}):\s*(.+)$")


def _clean_line(line):
    return re.sub(r"^[\s\-•*\d.)]+", "", line).strip()


def extract_structured(document: str, blocks: list) -> dict:
    out = {"document": document, "patient_information": {}, "conditions": [], "diagnoses": [],
           "medications": [], "lab_results": [], "procedures": [], "encounters": [], "follow_up": [],
           "report_date": ""}
    full = "\n".join(b["text"] for b in blocks)
    m = REPORT_DATE_RE.search(full)
    if m and first_date(m.group(1)):
        out["report_date"] = iso(first_date(m.group(1)))
    for b in blocks:
        sec = b["section"].lower()
        for key, names in SECTION_MAP.items():
            if sec not in names:
                continue
            for raw in b["text"].split("\n"):
                line = _clean_line(raw)
                if not line:
                    continue
                if key == "patient_information":
                    kv = KV_RE.match(line)
                    if kv:
                        out["patient_information"][kv.group(1).strip()] = kv.group(2).strip()
                    continue
                d = first_date(line)
                out[key].append({"text": line, "date": iso(d), "page": b["page"], "document": document})
    out["conditions"] = list(out["diagnoses"])
    out["lab_results"] = extract_labs(blocks, document)
    return out
