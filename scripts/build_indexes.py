"""Build FAISS + BM25 stores.
  python scripts/build_indexes.py --medical [--limit-medquad 5000]
  python scripts/build_indexes.py --patient            (indexes data/reports/*)"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from embeddings.embedding_service import build_embedder
from ingestion.chunker import chunk_text
from ingestion.document_processor import process_document
from ingestion.metadata import make_chunk_id, build_metadata
from retrieval.vector_store import VectorStore


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def medical_chunks(cfg, limit_medquad=None):
    proc, ct = Path(cfg["paths"]["processed_dir"]), cfg["chunking"]
    out = []
    mq = proc / "medquad.jsonl"
    if mq.exists():
        recs = read_jsonl(mq)[:limit_medquad] if limit_medquad else read_jsonl(mq)
        for r in recs:
            pieces = chunk_text(r["answer"], ct["medical_chunk_tokens"], ct["overlap_tokens"]) or [r["answer"]]
            for j, piece in enumerate(pieces, 1):
                out.append({"text": f"Q: {r['question']}\nA: {piece}", "metadata": build_metadata(
                    document=f"MedQuAD: {r['focus']}", page=None, section=r["question"],
                    chunk_id=f"medquad_{r['id']}_c{j}", source_type="medical_knowledge",
                    doc_type="qa", dataset="MedQuAD", extra={"url": r.get("url"), "source": r.get("source")})})
    pq = proc / "pubmedqa.jsonl"
    if pq.exists():
        for r in read_jsonl(pq):
            body = f"Question: {r['question']}\nContext: {r['context']}\nConclusion: {r['long_answer']} (expert label: {r['final_decision']})"
            for j, piece in enumerate(chunk_text(body, ct["medical_chunk_tokens"] * 2, ct["overlap_tokens"]), 1):
                out.append({"text": piece, "metadata": build_metadata(
                    document=f"PubMedQA: {r['pubid']}", page=None, section=r["question"],
                    chunk_id=f"pubmedqa_{r['pubid']}_c{j}", source_type="medical_knowledge",
                    doc_type="abstract", dataset="PubMedQA")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medical", action="store_true")
    ap.add_argument("--patient", action="store_true")
    ap.add_argument("--limit-medquad", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    emb = build_embedder(cfg)
    t0 = time.time()
    if a.medical:
        chunks = medical_chunks(cfg, a.limit_medquad)
        if not chunks:
            sys.exit("No medical data. Run scripts/download_data.py first.")
        vs = VectorStore(emb)
        vs.build(chunks, show_progress=True)
        vs.save(cfg["paths"]["medical_store"])
        print(f"Medical store: {len(chunks)} chunks -> {cfg['paths']['medical_store']} ({time.time() - t0:.0f}s)")
    if a.patient:
        rd = Path(cfg["paths"]["reports_dir"])
        files = [p for p in sorted(rd.glob("*")) if p.suffix.lower() in (".pdf", ".txt", ".docx")]
        if not files:
            sys.exit(f"No reports in {rd}. Run scripts/generate_reports.py first.")
        chunks, sdir = [], Path(cfg["paths"]["processed_dir"]) / "structured"
        sdir.mkdir(parents=True, exist_ok=True)
        c = cfg["chunking"]
        for f in files:
            d = process_document(f, c["chunk_tokens"], c["overlap_tokens"], dataset="Synthea-style (synthetic)")
            chunks += d.chunks
            (sdir / f"{f.stem}.json").write_text(json.dumps(d.structured, indent=1))
        vs = VectorStore(emb)
        vs.build(chunks, show_progress=True)
        vs.save(cfg["paths"]["patient_store"])
        print(f"Patient store: {len(files)} documents, {len(chunks)} chunks ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
