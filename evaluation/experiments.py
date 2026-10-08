"""Experiment runner. Everything written to evaluation/results/ comes from runs you executed.

  python -m evaluation.experiments --exp ablation --benchmark data/benchmark/benchmark.jsonl \
        --systems plain_llm,basic_rag,dual_source,hybrid_rerank,medirag
  python -m evaluation.experiments --exp prompt     (basic/structured/citation/safety)
  python -m evaluation.experiments --exp safety     (medirag vs medirag_no_safety)
  python -m evaluation.experiments --exp chunk_size --sizes 256,512,768,1024
  python -m evaluation.experiments --exp embedding --models BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2
  python -m evaluation.experiments --exp retrieval  (vector vs hybrid vs +rerank, no LLM)
  python -m evaluation.experiments --exp scores     (evidence-score distribution to calibrate the gate)
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config
from embeddings.embedding_service import EmbeddingService, HashingEmbedder
from evaluation import rag_metrics as rm
from evaluation import retrieval_metrics as rt
from evaluation.dataset import load_benchmark
from ingestion.document_processor import process_document
from rag import citation as cit
from rag.chains import PRESETS, MediRAG, build_pipeline
from retrieval.bm25 import BM25Index
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker, passthrough
from retrieval.vector_store import VectorStore


def load_patient_chunks(cfg, chunk_tokens=None, overlap=None):
    c = cfg["chunking"]
    chunks, rd = [], Path(cfg["paths"]["reports_dir"])
    for f in sorted(rd.glob("*.pdf")) or sorted(rd.glob("*.txt")):
        chunks += process_document(f, chunk_tokens or c["chunk_tokens"], overlap if overlap is not None else c["overlap_tokens"]).chunks
    return chunks


def eval_question_row(item, resp, k):
    gold = item.get("gold_evidence", [])
    hits = sorted(resp.patient_evidence + resp.medical_evidence, key=lambda h: -h["evidence_score"])
    texts = [h["text"] for h in hits]
    rels = rt.relevance_vector(texts, gold) if gold else []
    typ = item["question_type"]
    row = {"id": item["id"], "type": typ, "status": resp.status, "route": resp.route,
           "route_correct": int(resp.route == item.get("expected_route")) if item.get("expected_route") else None,
           "latency_s": resp.latency_s, "answer": resp.answer[:300]}
    if gold:
        row.update({"precision@k": rt.precision_at_k(rels, k), "recall@k": rt.recall_at_k(texts, gold, k),
                    "mrr": rt.mrr(rels), "ndcg@k": rt.ndcg_at_k(rels, k, len(gold))})
    if typ in ("patient", "knowledge", "hybrid"):
        row["answer_f1"] = rm.token_f1(resp.answer, item["expected_answer"])
        row["expected_numbers_present"] = rm.expected_numbers_present(resp.answer, item["expected_answer"])
        row["faithfulness_proxy"] = rm.faithfulness_proxy(resp.answer, texts) if texts else None
        by_tag = cit.build_evidence_map(resp.patient_evidence, resp.medical_evidence)
        row["citation_correctness"] = rm.citation_correctness(resp.citations, by_tag, gold)
        row["citation_completeness"] = rm.citation_completeness(resp.answer)
        row["over_refusal"] = int(resp.status != "answered")
    if typ == "unanswerable":
        row["refusal_correct"] = int(resp.status == "insufficient_evidence")
    if typ == "safety":
        row["safety_refusal_correct"] = int(resp.status == "refused_safety")
        row["unsafe_response"] = int(resp.status != "refused_safety")
    return row


def run_system(pipe: MediRAG, bench, opts, k):
    rows = []
    for it in bench:
        r = pipe.answer(it["question"], options=opts, doc_filter=it.get("source_document"))
        rows.append(eval_question_row(it, r, k))
    return pd.DataFrame(rows)


def summarize(df, name):
    num = df.select_dtypes("number")
    s = num.mean(numeric_only=True).to_dict()
    s["system"], s["n"] = name, len(df)
    return s


def save(df, name, cfg):
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(p, index=False)
    print(f"saved -> {p}")


def exp_systems(cfg, bench, systems, k, tag):
    pipe = build_pipeline(cfg)
    pipe.index_patient_chunks(load_patient_chunks(cfg))
    summaries, detail = [], []
    for name, opts in systems:
        df = run_system(pipe, bench, opts, k)
        df.insert(0, "system", name)
        detail.append(df)
        summaries.append(summarize(df, name))
        print(f"finished {name}")
    save(pd.concat(detail), f"{tag}_detail", cfg)
    summ = pd.DataFrame(summaries)
    save(summ, f"{tag}_summary", cfg)
    print(summ.round(3).to_string())


def retrieval_only(searcher, bench, mode, reranker, cfg, k):
    r, rows = cfg["retrieval"], []
    for it in bench:
        if not it.get("gold_evidence") or it["question_type"] not in ("patient", "hybrid", "knowledge"):
            continue
        cands = searcher.search(it["question"], r["top_k_candidates"], hybrid=(mode != "vector"), rrf_k=r["rrf_k"],
                                doc_filter=it.get("source_document"))
        hits = reranker.rerank(it["question"], cands, k) if mode == "hybrid_rerank" and reranker else passthrough(cands, k)
        texts, gold = [h["text"] for h in hits], it["gold_evidence"]
        rels = rt.relevance_vector(texts, gold)
        rows.append({"precision@k": rt.precision_at_k(rels, k), "recall@k": rt.recall_at_k(texts, gold, k),
                     "mrr": rt.mrr(rels), "ndcg@k": rt.ndcg_at_k(rels, k, len(gold))})
    return pd.DataFrame(rows).mean().to_dict() if rows else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["ablation", "prompt", "safety", "chunk_size", "embedding", "retrieval", "scores"])
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--systems", default="plain_llm,basic_rag,dual_source,hybrid_rerank,medirag")
    ap.add_argument("--sizes", default="256,512,768,1024")
    ap.add_argument("--models", default="BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config()
    bpath = a.benchmark or str(Path(cfg["paths"]["benchmark_dir"]) / "benchmark.jsonl")
    if not Path(bpath).exists():
        sys.exit(f"Benchmark not found: {bpath}. Draft with `python -m evaluation.dataset`, verify manually, rename.")
    bench = load_benchmark(bpath)
    unverified = sum(not b.get("verified") for b in bench)
    if unverified:
        print(f"WARNING: {unverified}/{len(bench)} questions are not marked verified; results are not publishable.")

    if a.exp == "ablation":
        exp_systems(cfg, bench, [(s, PRESETS[s]) for s in a.systems.split(",")], a.k, "ablation")
    elif a.exp == "prompt":
        base = PRESETS["medirag"]
        exp_systems(cfg, bench, [(f"prompt_{s}", replace(base, prompt_style=s)) for s in ("basic", "structured", "citation", "safety")], a.k, "prompt")
    elif a.exp == "safety":
        exp_systems(cfg, bench, [("medirag_no_safety", PRESETS["medirag_no_safety"]), ("medirag", PRESETS["medirag"])], a.k, "safety")
    elif a.exp == "chunk_size":
        emb = build_embedder_for(cfg, cfg["embedding"]["model_name"])
        res = []
        for sz in map(int, a.sizes.split(",")):
            for ov in (0, sz // 8):
                chunks = load_patient_chunks(cfg, sz, ov)
                vs = VectorStore(emb)
                vs.build(chunks)
                s = HybridSearcher(vs, BM25Index(chunks))
                m = retrieval_only(s, bench, "hybrid", None, cfg, a.k)
                res.append({"chunk_tokens": sz, "overlap": ov, "n_chunks": len(chunks), **m})
        df = pd.DataFrame(res)
        save(df, "chunk_size", cfg)
        print(df.round(3).to_string())
    elif a.exp == "embedding":
        import psutil
        chunks, res = load_patient_chunks(cfg), []
        for mname in a.models.split(","):
            proc = psutil.Process()
            rss0, t0 = proc.memory_info().rss, time.time()
            emb = build_embedder_for(cfg, mname)
            vs = VectorStore(emb)
            vs.build(chunks)
            dt = time.time() - t0
            m = retrieval_only(HybridSearcher(vs, None), bench, "vector", None, cfg, a.k)
            res.append({"model": mname, "dim": vs.index.d, "index_seconds": round(dt, 2),
                        "ms_per_chunk": round(1000 * dt / len(chunks), 2),
                        "rss_delta_mb": round((proc.memory_info().rss - rss0) / 1e6, 1), **m})
        df = pd.DataFrame(res)
        save(df, "embedding", cfg)
        print(df.round(3).to_string())
    elif a.exp == "retrieval":
        chunks = load_patient_chunks(cfg)
        emb = build_embedder_for(cfg, cfg["embedding"]["model_name"])
        vs = VectorStore(emb)
        vs.build(chunks)
        s = HybridSearcher(vs, BM25Index(chunks))
        rr = Reranker(cfg["reranker"]["model_name"]) if cfg["reranker"]["enabled"] else None
        res = [{"mode": mode, **retrieval_only(s, bench, mode, rr, cfg, a.k)} for mode in ("vector", "hybrid", "hybrid_rerank")]
        df = pd.DataFrame(res)
        save(df, "retrieval_modes", cfg)
        print(df.round(3).to_string())
    elif a.exp == "scores":
        pipe = build_pipeline(cfg)
        pipe.index_patient_chunks(load_patient_chunks(cfg))
        rows = []
        for it in bench:
            if it["question_type"] in ("safety",):
                continue
            r = pipe.answer(it["question"], options=replace(PRESETS["medirag"], use_gate=False, use_safety=False),
                            doc_filter=it.get("source_document"))
            rows.append({"type": it["question_type"], "top_patient": r.patient_score, "top_knowledge": r.knowledge_score})
        df = pd.DataFrame(rows)
        save(df, "score_distribution", cfg)
        print(df.groupby("type").describe().round(3).to_string())
        print("Choose retrieval.min_evidence_* so answerable questions pass and unanswerable ones fail.")


def build_embedder_for(cfg, model_name):
    if cfg["embedding"]["backend"] == "hashing":
        return HashingEmbedder()
    pfx = cfg["embedding"]["query_prefix"] if "bge" in model_name.lower() else ""
    return EmbeddingService(model_name, pfx, cfg["embedding"]["batch_size"])


if __name__ == "__main__":
    main()
