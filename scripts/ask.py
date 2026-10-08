"""CLI: python scripts/ask.py --report data/reports/SYN-0001.pdf -q "What was the hemoglobin?" """
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config
from ingestion.document_processor import process_document
from rag.chains import PRESETS, build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    ap.add_argument("-q", "--question", required=True)
    ap.add_argument("--preset", default="medirag", choices=list(PRESETS))
    a = ap.parse_args()
    cfg = load_config()
    pipe = build_pipeline(cfg)
    if a.report:
        c = cfg["chunking"]
        pipe.index_patient_chunks(process_document(a.report, c["chunk_tokens"], c["overlap_tokens"]).chunks)
    r = pipe.answer(a.question, options=PRESETS[a.preset])
    print(json.dumps({k: v for k, v in r.to_dict().items() if k not in ("patient_evidence", "medical_evidence")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
