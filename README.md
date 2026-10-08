# MediRAG
**Citation-grounded, dual-source Retrieval-Augmented Generation for explainable medical report question answering.**

> ⚠️ Educational research prototype. **Not a medical device. Does not diagnose, prescribe, or replace a clinician.**
> All patient data used here is **synthetic** (Synthea-style).

**Research question:** can a dual-source, citation-grounded RAG architecture improve factuality, relevance and
traceability of medical-report QA versus a plain LLM and basic RAG? The code lets you answer this empirically (ablations below).
It makes **no claim of novelty** and ships **no result numbers** — you generate them.

## Features (all implemented)
| Area | Implementation |
|---|---|
| Ingestion | PDF (PyMuPDF), TXT, DOCX → clean → section detection → structure-preserving chunking (table rows never split) → metadata (document, page, section, chunk_id, type, patient id, dataset) |
| Two knowledge bases | `patient_vector_store/` (uploaded report) and `medical_vector_store/` (MedQuAD + PubMedQA), kept separate |
| Retrieval | FAISS vectors + BM25 → Reciprocal Rank Fusion → cross-encoder reranker → top-K |
| Router | Deterministic rules: PATIENT_SPECIFIC / MEDICAL_KNOWLEDGE / HYBRID / UNSAFE_HIGH_RISK / UNANSWERABLE |
| Safety | Pre-generation risk classifier (informational vs clinical-decision vs high-risk), safe responses, post-generation output check |
| Anti-hallucination | retrieval grounding, strict prompts, evidence gate (low score or year absent from report → "insufficient evidence"), citation tags validated, numbers in the answer checked against evidence |
| Citations | `[P1]`/`[K1]` tags → document, page, section, chunk id + best supporting sentence |
| Query rewriting | follow-ups ("Is that normal?") resolved from chat history (rule-based; optional LLM) |
| Structured extraction | patient info, diagnoses, medications, encounters, labs (report-provided ranges only, never invented) |
| Lab dashboard / timeline | per-test table, status colours only when the report gives a range, dated timeline, descriptive trends |
| LLM | Hugging Face `transformers` (local), HF Inference API, or `extractive` (no LLM) |
| Evaluation | retrieval metrics, lexical RAG/citation/safety proxies, 6 experiment types |

**Not implemented (future work):** OCR/scanned PDFs, multimodal tables/images, agentic RAG, ChromaDB, MIMIC, RAGAS (LLM-judge; proxies provided instead), calibrated confidence.

## Setup (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add HF_TOKEN only if using hf_api / gated models
python -m pytest tests -q                              # MEDIRAG_TEST_MODE=1 python -m pytest tests -q  (no downloads)
```
Pick the LLM in `configs/config.yaml` (`llm.backend`): `transformers` (default Qwen2.5-1.5B-Instruct, CPU-OK but slow; use a 7–8B model on a GPU),
`hf_api` (needs `HF_TOKEN`), or `extractive`.

## Build the data and indexes
```bash
python scripts/download_data.py --medquad --pubmedqa        # prints REAL counts → data/processed/stats.json
python scripts/generate_reports.py --n 50                   # synthetic patient PDFs → data/reports/
#   or real Synthea CSVs:  python scripts/download_data.py --synthea && python scripts/generate_reports.py --source synthea --n 50
python scripts/build_indexes.py --medical                   # add --limit-medquad 3000 for a quick first run
python scripts/build_indexes.py --patient
```
Synthea has no lab reference ranges; those reports state "Not provided". The built-in generator assigns *typical adult intervals as a template* (document this limitation).

## Run
```bash
streamlit run app/streamlit_app.py                 # UI: Upload → Chat → Evidence → Labs → Timeline → Admin
python scripts/ask.py --report data/reports/SYN-0001.pdf -q "What was my hemoglobin?"
```
No-download demo of the plumbing only: `MEDIRAG_TEST_MODE=1 streamlit run app/streamlit_app.py` (hashing embedder + extractive answers — *not* representative quality).

## Evaluation workflow
1. `python -m evaluation.dataset` drafts 100-question skeleton (30/30/20/10/10) → `data/benchmark/benchmark_draft.jsonl`.
   **Drafts are auto-generated, not ground truth.** Review every entry (ideally two reviewers), set `"verified": true`, save as `benchmark.jsonl`.
   Knowledge questions drawn from MedQuAD are also in the index (leakage → inflated retrieval; hold some out or state it).
2. Calibrate the evidence gate: `python -m evaluation.experiments --exp scores`, then edit `retrieval.min_evidence_*`.
3. Run experiments (CSV → `evaluation/results/`, also viewable in the UI Admin page):
```bash
python -m evaluation.experiments --exp ablation       # plain LLM → basic RAG → dual-source → hybrid+rerank → MediRAG   (Exp 1-4)
python -m evaluation.experiments --exp prompt         # basic / structured / citation / safety prompt             (Exp 5)
python -m evaluation.experiments --exp safety         # with vs without safety layer                              (Exp 6)
python -m evaluation.experiments --exp embedding --models BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2   (Exp 7; add a biomedical model)
python -m evaluation.experiments --exp chunk_size --sizes 256,512,768,1024                                      (Exp 8)
python -m evaluation.experiments --exp retrieval      # vector vs hybrid vs hybrid+rerank, retrieval metrics only
```
Metrics: Precision/Recall@K, MRR, NDCG (gold *text spans*, robust to chunk-size changes); answer token-F1, expected-number match,
faithfulness proxy, citation correctness/completeness; routing accuracy; over-refusal, refusal accuracy, unsafe-response rate; latency; memory.
These are **lexical proxies**; validate against manual review before making claims, and report counts with the small-sample caveat (n=100).

## Results
*Intentionally empty.* Paste your own tables from `evaluation/results/` after running the experiments. Never report numbers from `MEDIRAG_TEST_MODE`.

## Limitations
Synthetic data (generalisation gap to real hospital PDFs); rule-based router and extractors (measure their accuracy); evidence score is relevance, not calibrated confidence;
small LLMs can ignore citation instructions (the validator flags, but cannot fix, this); no OCR; cosine-gate thresholds need calibration; MedQuAD answers may be missing for some NIH sources (filtered, counted in stats).

## Layout
`app/` UI · `ingestion/` · `extraction/` · `embeddings/` · `retrieval/` · `rag/` (prompts, citations, pipeline) · `llm/` · `safety/` · `evaluation/` · `scripts/` · `tests/` · `configs/config.yaml` (every experiment variable)

## Safety disclaimer
Informational/educational only. For diagnosis, treatment, medication changes or emergencies, contact a qualified healthcare professional or local emergency services.

## License
MIT (see LICENSE). Datasets keep their own licences (MedQuAD: NIH; PubMedQA: MIT; Synthea: Apache-2.0) — verify before redistribution.
