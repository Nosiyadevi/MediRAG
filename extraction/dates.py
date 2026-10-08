import re
from datetime import date, datetime

from dateutil import parser as dparser

_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
DATE_RE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b|\b{_MON}\s+\d{{1,2}},?\s+\d{{4}}\b|\b\d{{1,2}}\s+{_MON}\s+\d{{4}}\b",
    re.I)


def find_dates(text: str) -> list:
    return [m.group(0) for m in DATE_RE.finditer(text)]


def parse_date(s: str):
    s = s.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return datetime.strptime(s, "%Y-%m-%d").date()
        return dparser.parse(s, dayfirst=True).date()  # D/M/Y assumed for numeric formats
    except (ValueError, OverflowError):
        return None


def first_date(text: str):
    for d in find_dates(text):
        p = parse_date(d)
        if p:
            return p
    return None


def iso(d) -> str:
    return d.isoformat() if isinstance(d, date) else (d or "")
