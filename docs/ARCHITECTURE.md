# Architecture
```
Question ─► risk classifier ──(HIGH_RISK / CLINICAL_DECISION)──► safe response (no retrieval, no LLM)
        └─► query rewriter (history) ─► router
                 ├─ PATIENT_SPECIFIC ─► patient index
                 ├─ MEDICAL_KNOWLEDGE ► medical index (MedQuAD + PubMedQA)
                 └─ HYBRID ──────────► both
   each index: FAISS top-N + BM25 top-N ─► RRF ─► cross-encoder rerank ─► top-K
 ─► evidence gate (score threshold, year-not-in-report) ─► "insufficient evidence"
 ─► prompt ([P#] patient block, [K#] knowledge block) ─► HF LLM
 ─► post-check (unsafe phrasing), citation mapping, number-grounding check ─► answer + evidence + scores
```
Key decisions: UNSAFE is decided *before* generation; UNANSWERABLE *after* retrieval; evidence score = reranker sigmoid (or cosine without reranker), reported as relevance, not confidence.
Evidence gate limitation: without the reranker, lexical-only hits have low cosine, so only the best score gates (no per-hit filtering).

Data flow for upload: file → `pdf_loader` → `document_processor.clean_text/detect_sections` → `chunker.chunk_blocks` → `VectorStore` + `BM25Index`; in parallel `medical_entities.extract_structured` → labs/timeline pages.
