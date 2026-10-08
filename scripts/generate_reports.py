"""Create patient reports (PDF/TXT) for the patient-side corpus.

Modes
  --source builtin  Synthea-style synthetic patients generated here (default; no download needed).
                    Reference ranges are TEMPLATE-ASSIGNED typical adult intervals (documented limitation).
  --source synthea  Convert Synthea CSV exports (patients/encounters/conditions/medications/observations).
                    Synthea has NO reference ranges, so none are written (system will say so).
All data is SYNTHETIC. Run:  python scripts/generate_reports.py --n 50
"""
import argparse
import random
import sys
import textwrap
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# name, unit, (low_f, high_f), (low_m, high_m), decimals ; low None => upper limit only
LABS = [
    ("Hemoglobin", "g/dL", (12.0, 15.5), (13.5, 17.5), 1),
    ("WBC", "10^3/uL", (4.0, 11.0), (4.0, 11.0), 1),
    ("Platelets", "10^3/uL", (150, 450), (150, 450), 0),
    ("Glucose (Fasting)", "mg/dL", (70, 99), (70, 99), 0),
    ("HbA1c", "%", (4.0, 5.6), (4.0, 5.6), 1),
    ("Creatinine", "mg/dL", (0.6, 1.1), (0.7, 1.3), 2),
    ("Sodium", "mmol/L", (135, 145), (135, 145), 0),
    ("Potassium", "mmol/L", (3.5, 5.1), (3.5, 5.1), 1),
    ("Total Cholesterol", "mg/dL", (None, 200), (None, 200), 0),
]
CONDITIONS = [
    ("Hypertension", "Lisinopril 10 mg once daily"),
    ("Type 2 diabetes mellitus", "Metformin 500 mg twice daily"),
    ("Iron deficiency anemia", "Ferrous sulfate 325 mg once daily"),
    ("Hyperlipidemia", "Atorvastatin 20 mg once daily"),
    ("Community-acquired pneumonia", "Amoxicillin-clavulanate 875/125 mg twice daily"),
    ("Hypothyroidism", "Levothyroxine 50 mcg once daily"),
    ("Asthma", "Albuterol inhaler as needed"),
]
VISITS = ["Outpatient visit: routine review", "Emergency department visit", "Hospital admission",
          "Follow-up clinic visit", "Hospital discharge"]
NAMES = ["Asha", "Ravi", "Meena", "Karthik", "Priya", "Arjun", "Divya", "Suresh", "Lakshmi", "Vikram"]


def fmt(v, d):
    return f"{v:.{d}f}"


def lab_value(rng, spec, abnormal):
    _, _, fr, mr, dec = spec
    lo, hi = rng
    if lo is None:
        return rng_val(150, 199, dec) if not abnormal else rng_val(201, 260, dec)
    if not abnormal:
        return rng_val(lo, hi, dec)
    return rng_val(lo * 0.65, lo * 0.95, dec) if rng_choice() else rng_val(hi * 1.05, hi * 1.4, dec)


_R = random.Random(0)


def rng_val(a, b, dec):
    return round(_R.uniform(a, b), dec)


def rng_choice():
    return _R.random() < 0.5


def render_report(pid, name, age, sex, visit_dates, diagnoses, meds, labs_by_date, course, follow_up):
    L = ["DISCHARGE SUMMARY",
         "SYNTHETIC DOCUMENT - generated for academic research. Not real patient data.", "",
         "PATIENT INFORMATION", f"Patient ID: {pid}", f"Name: {name}", f"Age: {age}", f"Sex: {sex}",
         f"Report Date: {visit_dates[-1].isoformat()}", "", "DIAGNOSES"]
    L += [f"- {d}" for d in diagnoses] + ["", "MEDICATIONS"] + [f"- {m}" for m in meds] + ["", "ENCOUNTERS"]
    L += [f"{d.isoformat()} - {VISITS[min(i, len(VISITS) - 1)]}" for i, d in enumerate(visit_dates)]
    L += ["", "HOSPITAL COURSE"] + textwrap.wrap(course, 88) + ["", "LABORATORY RESULTS"]
    for d, rows in labs_by_date.items():
        L += [f"Collected: {d.isoformat()}", "Test | Result | Reference Range | Flag"]
        for test, val, unit, rng, flag in rows:
            L.append(f"{test} | {val} {unit} | {rng if rng else 'Not provided'} | {flag}")
        L.append("")
    L += ["DISCHARGE PLAN"] + [f"- {f}" for f in follow_up]
    return "\n".join(L)


def build_builtin(i, rnd):
    global _R
    _R = rnd
    sex = rnd.choice(["Female", "Male"])
    age = rnd.randint(25, 82)
    name = f"{rnd.choice(NAMES)} Synthetic{i:04d}"
    conds = rnd.sample(CONDITIONS, rnd.randint(1, 3))
    nvis = rnd.randint(1, 3)
    last = date(2023, 1, 1) + timedelta(days=rnd.randint(150, 700))
    dates = sorted({last - timedelta(days=60 * k) for k in range(nvis)})
    abn = {s[0]: rnd.random() < 0.35 for s in LABS}
    labs_by_date = {}
    for d in dates:
        rows = []
        for s in LABS:
            lo, hi = s[2] if sex == "Female" else s[3]
            v = lab_value((lo, hi), s, abn[s[0]])
            rng = f"<{hi}" if lo is None else f"{fmt(lo, s[4])} - {fmt(hi, s[4])}"
            flag = ("H" if v > hi else "L" if (lo is not None and v < lo) else "")
            rows.append((s[0], fmt(v, s[4]), s[1], rng, flag))
        labs_by_date[d] = rows
    course = (f"The patient has a history of {', '.join(c[0].lower() for c in conds)}. "
              f"Across {nvis} recorded visit(s) the patient was evaluated and laboratory tests were obtained. "
              "The patient was stable at the time of the last encounter.")
    follow = ["Follow-up with primary care physician in 2 weeks.", "Repeat laboratory tests in 4 weeks."]
    pid = f"SYN-{i:04d}"
    return pid, render_report(pid, name, age, sex, dates, [c[0] for c in conds], [c[1] for c in conds],
                              labs_by_date, course, follow)


def write_pdf(text, path):
    import fitz
    doc, lines, per = fitz.open(), text.split("\n"), 54
    for s in range(0, len(lines), per):
        page, y = doc.new_page(), 50
        for line in lines[s:s + per]:
            page.insert_text((40, y), line, fontname="cour", fontsize=9)
            y += 13.5
    doc.save(str(path))
    doc.close()


def from_synthea(src: Path, n, out: Path, fmt_):
    import pandas as pd
    pats = pd.read_csv(src / "patients.csv").head(n)
    enc, con = pd.read_csv(src / "encounters.csv"), pd.read_csv(src / "conditions.csv")
    med, obs = pd.read_csv(src / "medications.csv"), pd.read_csv(src / "observations.csv")
    for k, p in pats.iterrows():
        pid = p["Id"]
        e = enc[enc.PATIENT == pid].sort_values("START").tail(5)
        dates = [pd.to_datetime(x).date() for x in e.START] or [date.today()]
        o = obs[(obs.PATIENT == pid) & (obs.TYPE == "numeric")]
        if "CATEGORY" in o.columns:
            o = o[o.CATEGORY == "laboratory"]
        o = o.dropna(subset=["UNITS"])
        labs_by_date = {}
        for d, g in o.groupby(pd.to_datetime(o.DATE).dt.date):
            labs_by_date[d] = [(r.DESCRIPTION, r.VALUE, r.UNITS, None, "") for r in g.itertuples()][:15]
        labs_by_date = dict(list(sorted(labs_by_date.items()))[-3:])
        age = (dates[-1] - pd.to_datetime(p["BIRTHDATE"]).date()).days // 365
        text = render_report(f"SYN-{k:04d}", f"{p.get('FIRST', '')} {p.get('LAST', '')}".strip(), age,
                             {"F": "Female", "M": "Male"}.get(p.get("GENDER"), "Unknown"), dates,
                             sorted(set(con[con.PATIENT == pid].DESCRIPTION))[:8] or ["None recorded"],
                             sorted(set(med[med.PATIENT == pid].DESCRIPTION))[:8] or ["None recorded"],
                             labs_by_date, "Record converted from Synthea synthetic data.", ["Routine follow-up."])
        emit(f"SYN-{k:04d}", text, out, fmt_)


def emit(pid, text, out: Path, fmt_):
    out.mkdir(parents=True, exist_ok=True)
    if fmt_ in ("txt", "both"):
        (out / f"{pid}.txt").write_text(text, encoding="utf-8")
    if fmt_ in ("pdf", "both"):
        write_pdf(text, out / f"{pid}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["builtin", "synthea"], default="builtin")
    ap.add_argument("--synthea-dir", default=str(ROOT / "data" / "synthea"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--format", choices=["pdf", "txt", "both"], default="pdf")
    ap.add_argument("--out", default=str(ROOT / "data" / "reports"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    if a.source == "builtin":
        rnd = random.Random(a.seed)
        for i in range(1, a.n + 1):
            pid, text = build_builtin(i, rnd)
            emit(pid, text, out, a.format)
    else:
        csv_dir = next((p.parent for p in Path(a.synthea_dir).rglob("patients.csv")), None)
        if csv_dir is None:
            sys.exit("patients.csv not found. Run scripts/download_data.py --synthea first.")
        from_synthea(csv_dir, a.n, out, a.format)
    print(f"Wrote {a.n} reports to {out} (all SYNTHETIC)")


if __name__ == "__main__":
    main()
