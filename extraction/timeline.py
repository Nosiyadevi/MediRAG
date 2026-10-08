"""Patient timeline + descriptive lab trends (no medical interpretation)."""
import pandas as pd

from .dates import parse_date

EVENT_KEYS = {"diagnoses": "Diagnosis", "medications": "Medication", "procedures": "Procedure",
              "encounters": "Encounter", "follow_up": "Follow-up"}


def build_timeline(structured_list: list) -> pd.DataFrame:
    events = []
    for s in structured_list:
        doc = s["document"]
        for key, label in EVENT_KEYS.items():
            for it in s.get(key, []):
                if it.get("date"):
                    events.append({"Date": it["date"], "Type": label, "Event": it["text"],
                                   "Source": f"{doc} — Page {it['page']}"})
        by_date = {}
        for l in s.get("lab_results", []):
            if l["date"]:
                by_date.setdefault(l["date"], []).append(l)
        for d, ls in by_date.items():
            events.append({"Date": d, "Type": "Laboratory", "Event": f"Lab tests ({len(ls)} results): " +
                           ", ".join(l["test"] for l in ls[:6]) + ("..." if len(ls) > 6 else ""),
                           "Source": f"{doc} — Page {ls[0]['source_page']}"})
    df = pd.DataFrame(events, columns=["Date", "Type", "Event", "Source"])
    if not df.empty:
        df["_d"] = df["Date"].map(parse_date)
        df = df.sort_values("_d").drop(columns="_d").reset_index(drop=True)
    return df


def lab_trend(labs: list, test: str):
    pts = sorted([l for l in labs if l["test"].lower() == test.lower() and l["date"]], key=lambda l: l["date"])
    if len(pts) < 2:
        return pts, "Fewer than two dated values; no trend can be described."
    a, b = pts[0], pts[-1]
    diff = b["value"] - a["value"]
    word = "increased" if diff > 0 else "decreased" if diff < 0 else "did not change"
    txt = (f"Observed: {test} {word} from {a['value']} {a['unit']} ({a['date']}) to "
           f"{b['value']} {b['unit']} ({b['date']})" + (f", a change of {diff:+.2f}." if diff else ".") +
           " This is a description of recorded values, not a medical interpretation.")
    return pts, txt
