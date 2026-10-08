"""MediRAG pipeline: safety -> rewrite -> route -> retrieve (hybrid+rerank) -> gate -> generate -> verify."""
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from embeddings.embedding_service import build_embedder
from llm.model import LLMClient
from rag import citation as cit
from rag.prompts import build_messages, format_context
from rag.query_rewriter import rewrite_query
from retrieval.bm25 import BM25Index
from retrieval.hybrid_search import HybridSearcher
from retrieval.query_router import Route, route_query
from retrieval.reranker import Reranker, passthrough
from retrieval.vector_store import VectorStore
from safety import guardrails
from safety.risk_classifier import Risk, classify_risk

log = logging.getLogger(__name__)


@dataclass
class PipelineOptions:
    use_retrieval: bool = True
    use_router: bool = True
    use_knowledge: bool = True
    use_hybrid: bool = True
    use_reranker: bool = True
    use_safety: bool = True
    use_gate: bool = True
    use_rewrite: bool = True
    prompt_style: str = "safety"


PRESETS = {
    "plain_llm": PipelineOptions(use_retrieval=False, use_router=False, use_knowledge=False, use_hybrid=False,
                                 use_reranker=False, use_safety=False, use_gate=False, use_rewrite=False, prompt_style="plain"),
    "basic_rag": PipelineOptions(use_router=False, use_knowledge=False, use_hybrid=False, use_reranker=False,
                                 use_safety=False, use_gate=False, use_rewrite=False, prompt_style="basic"),
    "dual_source": PipelineOptions(use_hybrid=False, use_reranker=False, use_safety=False, use_gate=False,
                                   use_rewrite=False, prompt_style="structured"),
    "hybrid_rerank": PipelineOptions(use_safety=False, use_gate=False, use_rewrite=False, prompt_style="structured"),
    "medirag": PipelineOptions(),
    "medirag_no_safety": PipelineOptions(use_safety=False),
}


@dataclass
class RAGResponse:
    question: str
    rewritten_question: str = ""
    answer: str = ""
    status: str = "answered"   # answered | refused_safety | insufficient_evidence | error
    route: str = ""
    route_reason: str = ""
    risk: str = Risk.INFORMATIONAL.value
    patient_evidence: list = field(default_factory=list)
    medical_evidence: list = field(default_factory=list)
    citations: list = field(default_factory=list)
    evidence_score: Optional[float] = None
    patient_score: Optional[float] = None
    knowledge_score: Optional[float] = None
    warnings: list = field(default_factory=list)
    latency_s: float = 0.0
    prompt_style: str = ""
    disclaimer: str = guardrails.DISCLAIMER

    def to_dict(self):
        return asdict(self)


def extractive_answer(question, p_hits, k_hits):
    parts = []
    for i, h in enumerate(p_hits[:1], start=1):
        sents = cit.best_supporting_sentences(question, h["text"], 2) or [(h["text"].split("\n")[0], 0)]
        parts.append("According to the uploaded report: " + " ".join(s for s, _ in sents) + " [P1]")
    for h in k_hits[:1]:
        body = h["text"].split("A:", 1)[-1].strip()
        sents = cit.best_supporting_sentences(question, body, 2) or [(s, 0) for s in cit.split_sentences(body)[:2]]
        parts.append("General medical knowledge: " + " ".join(s for s, _ in sents) + " [K1]")
    return "\n\n".join(parts)


class MediRAG:
    def __init__(self, cfg, embedder, llm, reranker=None):
        self.cfg, self.embedder, self.llm, self.reranker = cfg, embedder, llm, reranker
        self.patient: Optional[HybridSearcher] = None
        self.medical: Optional[HybridSearcher] = None

    # ---- index management
    def load_medical(self, directory=None) -> bool:
        d = Path(directory or self.cfg["paths"]["medical_store"])
        if not (d / "chunks.jsonl").exists():
            log.warning("Medical store not found at %s (run scripts/build_indexes.py --medical)", d)
            return False
        store = VectorStore.load(d, self.embedder)
        self.medical = HybridSearcher(store, BM25Index(store.chunks))
        return True

    def index_patient_chunks(self, chunks, save_dir=None, show_progress=False):
        store = VectorStore(self.embedder)
        store.build(chunks, show_progress)
        self.patient = HybridSearcher(store, BM25Index(chunks))
        if save_dir:
            store.save(save_dir)

    @property
    def has_patient(self):
        return self.patient is not None and len(self.patient.store) > 0

    # ---- retrieval
    def _retrieve(self, searcher, query, opts, doc_filter):
        r = self.cfg["retrieval"]
        cands = searcher.search(query, r["top_k_candidates"], opts.use_hybrid, r["rrf_k"], doc_filter)
        if opts.use_reranker and self.reranker is not None:
            return self.reranker.rerank(query, cands, r["top_k_final"]), r["min_evidence_reranked"], True
        return passthrough(cands, r["top_k_final"]), r["min_evidence_cosine"], False

    def _missing_years(self, question, doc_filter):
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        if not years or not self.has_patient:
            return []
        corpus = " ".join(c["text"] for c in self.patient.store.chunks
                          if not doc_filter or c["metadata"]["document"] == doc_filter)
        return [y for y in years if y not in corpus]

    # ---- main entry
    def answer(self, question, history=None, options: Optional[PipelineOptions] = None, doc_filter=None) -> RAGResponse:
        t0 = time.time()
        opts, history = options or PRESETS["medirag"], history or []
        resp = RAGResponse(question=question, prompt_style=opts.prompt_style)
        try:
            self._answer(resp, question, history, opts, doc_filter)
        except Exception as e:  # never crash the UI
            log.exception("pipeline error")
            resp.status, resp.answer = "error", f"An error occurred while generating the answer: {e}"
        resp.latency_s = round(time.time() - t0, 3)
        return resp

    def _answer(self, resp, question, history, opts, doc_filter):
        risk = classify_risk(question)
        resp.risk = risk.level.value
        if opts.use_safety and risk.level != Risk.INFORMATIONAL:
            resp.route, resp.route_reason = Route.UNSAFE.value, f"risk category: {risk.category}"
            resp.status, resp.answer = "refused_safety", guardrails.safety_response(risk.category)
            return

        q = rewrite_query(question, history, self.llm, self.cfg["rewriting"]["use_llm"]) if opts.use_rewrite else question
        resp.rewritten_question = q

        if not opts.use_retrieval:
            system, user = build_messages("plain", "", "", question)
            resp.route = "NO_RETRIEVAL"
            resp.answer = self.llm.generate(system, user)
            return

        # routing
        if opts.use_router:
            d = route_query(q, self.has_patient)
            route, resp.route_reason = d.route, d.reason
        else:
            route = Route.HYBRID if opts.use_knowledge else Route.PATIENT
            resp.route_reason = "router disabled"
        resp.route = route.value
        if route == Route.UNANSWERABLE:
            resp.status, resp.answer = "insufficient_evidence", guardrails.INSUFFICIENT
            resp.warnings.append(resp.route_reason)
            return

        use_p = route in (Route.PATIENT, Route.HYBRID) and self.has_patient
        use_k = opts.use_knowledge and route in (Route.KNOWLEDGE, Route.HYBRID) and self.medical is not None
        p_hits, k_hits, p_thr, k_thr, p_rr, k_rr = [], [], 0.0, 0.0, False, False
        if use_p:
            p_hits, p_thr, p_rr = self._retrieve(self.patient, q, opts, doc_filter)
        if use_k:
            k_hits, k_thr, k_rr = self._retrieve(self.medical, q, opts, None)

        # evidence gate (hallucination safeguard #4/#5)
        if opts.use_gate:
            missing = self._missing_years(question, doc_filter) if use_p else []
            p_ok = bool(p_hits) and max(h["evidence_score"] for h in p_hits) >= p_thr and not missing
            k_ok = bool(k_hits) and max(h["evidence_score"] for h in k_hits) >= k_thr
            if missing:
                resp.warnings.append(f"Year(s) {', '.join(missing)} do not appear in the report.")
            ok = {Route.PATIENT: p_ok, Route.KNOWLEDGE: k_ok, Route.HYBRID: p_ok or k_ok}[route]
            if not ok:
                resp.status, resp.answer = "insufficient_evidence", guardrails.INSUFFICIENT
                resp.patient_evidence, resp.medical_evidence = p_hits, k_hits
                return
            if route == Route.HYBRID and use_p and not p_ok:
                p_hits = []
                resp.warnings.append("No supporting evidence found in the uploaded report; answer uses general knowledge only.")
            if use_k and not k_ok:
                k_hits = []
            # per-hit filtering only for reranker scores; cosine would wrongly drop lexical (BM25) matches
            if p_rr:
                p_hits = [h for h in p_hits if h["evidence_score"] >= p_thr]
            if k_rr:
                k_hits = [h for h in k_hits if h["evidence_score"] >= k_thr]

        resp.patient_evidence, resp.medical_evidence = p_hits, k_hits
        resp.patient_score = max((h["evidence_score"] for h in p_hits), default=None)
        resp.knowledge_score = max((h["evidence_score"] for h in k_hits), default=None)
        resp.evidence_score = resp.patient_score if resp.patient_score is not None else resp.knowledge_score

        # generation
        if self.llm.backend == "extractive":
            answer = extractive_answer(q, p_hits, k_hits)
        else:
            system, user = build_messages(opts.prompt_style, format_context(p_hits, "P"), format_context(k_hits, "K"), q)
            answer = self.llm.generate(system, user)

        # post-checks: safety, citations, grounding
        if opts.use_safety:
            answer, flags = guardrails.post_check(answer)
            if flags:
                resp.warnings.append("Answer replaced by safety layer (unsafe output pattern).")
        emap = cit.build_evidence_map(p_hits, k_hits)
        tags = cit.extract_cited_tags(answer)
        for t in [t for t in tags if t not in emap]:
            answer = answer.replace(f"[{t}]", "")
            resp.warnings.append(f"Removed invalid citation tag [{t}].")
        resp.citations = [cit.citation_entry(t, emap[t], cit.claim_for_tag(answer, t, q)) for t in tags if t in emap]
        if opts.prompt_style in ("citation", "safety") or self.llm.backend == "extractive":
            bad = cit.validate_grounding(answer, emap)
            if bad:
                resp.warnings.append(f"Numbers in the answer not found in the evidence: {', '.join(bad)}")
            if p_hits and cit.uncited_numeric_sentences(answer):
                resp.warnings.append("Some numeric statements are not cited.")
        resp.answer = answer.strip()


def build_pipeline(cfg, load_medical=True) -> MediRAG:
    embedder = build_embedder(cfg)
    reranker = Reranker(cfg["reranker"]["model_name"]) if cfg["reranker"]["enabled"] else None
    pipe = MediRAG(cfg, embedder, LLMClient.from_config(cfg), reranker)
    if load_medical:
        pipe.load_medical()
    return pipe
