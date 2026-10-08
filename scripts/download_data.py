"""Download + preprocess MedQuAD, PubMedQA (and optionally Synthea sample CSVs).
  python scripts/download_data.py --medquad --pubmedqa
Writes data/processed/{medquad,pubmedqa}.jsonl and data/processed/stats.json (real counts)."""
import argparse
import json
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"
SYNTHEA_URL = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip"


def medquad():
    dest = ROOT / "data" / "medquad"
    if not any(dest.glob("*_QA")):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/abachaa/MedQuAD", str(dest)], check=True)
    recs, empty, bad, docs = [], 0, 0, 0
    for xml in sorted(dest.rglob("*.xml")):
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            bad += 1
            continue
        docs += 1
        focus = (root.findtext("Focus") or "").strip()
        for qa in root.iter("QAPair"):
            q, a = qa.find("Question"), qa.find("Answer")
            qt, at = (q.text or "").strip() if q is not None else "", (a.text or "").strip() if a is not None else ""
            if not qt or not at:
                empty += 1
                continue
            recs.append({"id": f"{xml.stem}_{len(recs)}", "focus": focus, "question": qt, "answer": at,
                         "qtype": q.attrib.get("qtype", ""), "source": root.attrib.get("source", xml.parent.name),
                         "url": root.attrib.get("url", "")})
    write(PROC / "medquad.jsonl", recs)
    return {"medquad_xml_files": docs, "medquad_qa_pairs": len(recs), "medquad_empty_answers_skipped": empty,
            "medquad_unparseable_files": bad}


def pubmedqa():
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    recs = []
    for r in ds:
        ctx = " ".join(r["context"]["contexts"])
        recs.append({"pubid": str(r["pubid"]), "question": r["question"], "context": ctx,
                     "long_answer": r["long_answer"], "final_decision": r["final_decision"]})
    write(PROC / "pubmedqa.jsonl", recs)
    return {"pubmedqa_labeled_records": len(recs)}


def synthea():
    dest = ROOT / "data" / "synthea"
    dest.mkdir(parents=True, exist_ok=True)
    z = dest / "synthea_sample.zip"
    urllib.request.urlretrieve(SYNTHEA_URL, z)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    return {"synthea_csv_dir": str(next(dest.rglob("patients.csv")).parent)}


def write(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    for k in ("medquad", "pubmedqa", "synthea"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if not (a.medquad or a.pubmedqa or a.synthea):
        a.medquad = a.pubmedqa = True
    stats_path = PROC / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if a.medquad:
        stats.update(medquad())
    if a.pubmedqa:
        stats.update(pubmedqa())
    if a.synthea:
        stats.update(synthea())
    PROC.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
