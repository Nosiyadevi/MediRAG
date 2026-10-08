"""Rule-based lab result extraction. Reference ranges are NEVER invented:
only ranges printed in the report are kept; otherwise status says so."""
import re

import pandas as pd

from .dates import find_dates, first_date, iso, parse_date

LAB_SECTIONS = {"laboratory results", "lab results", "laboratory findings", "investigations", "laboratory", "labs"}
HEADER_WORDS = {"test", "test name", "name", "component", "analyte", "investigation", "parameter"}
FLAGS = {"h": "High", "l": "Low", "high": "High", "low": "Low", "normal": "Normal", "abnormal": "Abnormal",
         "critical": "Critical"}
_UNIT = r"(?:10\^\d+/)?[A-Za-zµ%][^\s]*"
RANGE_RE = re.compile(r"^\(?\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*\)?$")
LIMIT_RE = re.compile(r"^(<=|>=|<|>|≤|≥)\s*(\d+(?:\.\d+)?)$")
VALUE_UNIT_RE = re.compile(rf"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT})?$")
COLON_RE = re.compile(
    rf"^(?P<test>[A-Za-z][A-Za-z0-9 ,()/%\-]{{1,40}}?):\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT})?"
    r"\s*(?:\((?:ref(?:erence)?[^0-9<>]*)?(?P<range>[^)]*)\))?")
SPLIT_RE = re.compile(r"\s*\|\s*|\t+|\s{2,}")
DATE_LINE_RE = re.compile(r"^(?:collected|collection date|specimen date|test date|sample date|date)\s*[:\-]\s*(.+)$", re.I)
REPORT_DATE_RE = re.compile(r"(?:report date|date of report|discharge date)\s*[:\-]\s*(.+)", re.I)


def _is_range(s):
    return bool(RANGE_RE.match(s) or LIMIT_RE.match(s))


def _range_bounds(rng):
    if not rng:
        return None, None, None
    m = RANGE_RE.match(rng.strip())
    if m:
        return float(m.group(1)), float(m.group(2)), None
    m = LIMIT_RE.match(rng.strip())
    if m:
        return None, None, (m.group(1), float(m.group(2)))
    return None, None, None


def derive_status(value, rng, flag):
    low, high, limit = _range_bounds(rng)
    if low is not None:
        return "Below report range" if value < low else "Above report range" if value > high else "Within report range"
    if limit:
        op, lim = limit
        ok = {"<": value < lim, "<=": value <= lim, "≤": value <= lim,
              ">": value > lim, ">=": value >= lim, "≥": value >= lim}[op]
        return "Within report range" if ok else ("Above report range" if op in "<≤" or op == "<=" else "Below report range")
    if flag:
        return f"Report flag: {flag}"
    return "No reference range in report"


def _parse_fields(fields):
    if len(fields) < 2:
        return None
    test = fields[0]
    if test.lower() in HEADER_WORDS or not re.match(r"[A-Za-z]", test):
        return None
    m = VALUE_UNIT_RE.match(fields[1])
    if not m:
        return None
    value, unit = float(m.group("value")), m.group("unit") or ""
    rest = fields[2:]
    if not unit and rest and not _is_range(rest[0]) and rest[0].lower() not in FLAGS and re.match(_UNIT + "$", rest[0]):
        unit = rest.pop(0)
    rng = flag = None
    for r in rest:
        if _is_range(r):
            rng = r
        elif r.lower() in FLAGS:
            flag = FLAGS[r.lower()]
    return test, value, unit, rng, flag


def parse_lab_line(line: str, in_lab_section: bool):
    line = line.strip().lstrip("-•* ").strip()
    if not line or DATE_LINE_RE.match(line):
        return None
    parsed = None
    if "|" in line or "\t" in line or re.search(r"\s{2,}", line):
        parsed = _parse_fields([f.strip() for f in SPLIT_RE.split(line) if f.strip()])
    if parsed is None:
        m = COLON_RE.match(line)
        if m:
            rng = (m.group("range") or "").strip() or None
            parsed = (m.group("test").strip(), float(m.group("value")), m.group("unit") or "",
                      rng if rng and _is_range(rng) else None, None)
    if parsed is None:
        return None
    test, value, unit, rng, flag = parsed
    if not in_lab_section and not ("/" in unit or "%" in unit):
        return None  # avoid parsing medication doses like "10 mg" outside lab sections
    return test, value, unit, rng, flag


def extract_labs(blocks: list, document: str) -> list:
    full = "\n".join(b["text"] for b in blocks)
    m = REPORT_DATE_RE.search(full)
    doc_date = first_date(m.group(1)) if m else None
    labs, current = [], None
    for b in blocks:
        in_lab = b["section"].lower() in LAB_SECTIONS
        for line in b["text"].split("\n"):
            dm = DATE_LINE_RE.match(line.strip())
            if dm:
                d = first_date(dm.group(1))
                if d:
                    current = d
                continue
            r = parse_lab_line(line, in_lab)
            if not r:
                continue
            test, value, unit, rng, flag = r
            labs.append({
                "test": test, "value": value, "unit": unit, "reference_range": rng,
                "date": iso(current or doc_date), "source_page": b["page"], "document": document,
                "section": b["section"], "report_flag": flag,
                "status": derive_status(value, rng, flag), "raw_line": line.strip(),
            })
    return labs


def labs_to_dataframe(labs: list) -> pd.DataFrame:
    cols = ["Test", "Result", "Unit", "Reference Range", "Date", "Source", "Status"]
    if not labs:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{
        "Test": l["test"], "Result": l["value"], "Unit": l["unit"],
        "Reference Range": l["reference_range"] or "Not provided in report", "Date": l["date"] or "Unknown",
        "Source": f"{l['document']} — Page {l['source_page']}", "Status": l["status"]} for l in labs], columns=cols)
