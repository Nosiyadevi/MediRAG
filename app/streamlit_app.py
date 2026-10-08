"""MediRAG Streamlit UI (v2).  Run from the project root:  streamlit run app/streamlit_app.py"""
import html
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config import load_config
from extraction.lab_extractor import labs_to_dataframe
from extraction.timeline import build_timeline, lab_trend
from ingestion.document_processor import process_document
from rag import citation as cit
from rag.chains import PRESETS, build_pipeline
from safety.guardrails import DISCLAIMER

st.set_page_config(page_title="MediRAG — Medical Report Q&A", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.block-container{padding-top:1.6rem;max-width:1180px;}
footer{visibility:hidden;}
section[data-testid="stSidebar"]{background:#F0F7F8;border-right:1px solid #DDEBED;}
.hero{background:linear-gradient(135deg,#0B6169 0%,#0E7C86 45%,#3FA9B5 100%);color:#fff;padding:1.7rem 2rem;
      border-radius:18px;margin-bottom:1.1rem;box-shadow:0 6px 18px rgba(14,124,134,.22);}
.hero h1{color:#fff;margin:0;font-size:2.1rem;font-weight:700;letter-spacing:-.3px;}
.hero p{color:#E3F5F7;margin:.45rem 0 0;font-size:1.02rem;}
.hero.small{padding:1.1rem 1.6rem;} .hero.small h1{font-size:1.6rem;}
.card{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:1rem 1.2rem;
      box-shadow:0 1px 3px rgba(15,23,42,.06);height:100%;}
.card h4{margin:0 0 .35rem;color:#0B6169;font-size:1.02rem;}
.card p{margin:0;color:#475569;font-size:.9rem;line-height:1.45;}
.tile{background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:.85rem 1rem;text-align:center;
      box-shadow:0 1px 3px rgba(15,23,42,.05);}
.tile .v{font-size:1.7rem;font-weight:700;color:#0B6169;line-height:1.15;}
.tile .l{font-size:.78rem;color:#64748B;text-transform:uppercase;letter-spacing:.5px;}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.74rem;font-weight:600;margin:0 6px 4px 0;}
.pill-teal{background:#DDF3F5;color:#0B6169;} .pill-purple{background:#EBE5F8;color:#5B3FA6;}
.pill-blue{background:#E0EDFF;color:#1D4ED8;} .pill-amber{background:#FFF0D1;color:#92580A;}
.pill-red{background:#FDE2E2;color:#B42323;} .pill-green{background:#DCF5E4;color:#17703A;}
.pill-gray{background:#EDF1F5;color:#475569;}
.ev{border-left:4px solid #0E7C86;background:#F6FAFB;border-radius:8px;padding:.7rem 1rem;margin-bottom:.75rem;}
.ev.k{border-left-color:#7C5CBF;background:#FAF8FD;}
.ev .src{font-size:.84rem;color:#334155;font-weight:600;}
.ev .meta{font-size:.74rem;color:#94A3B8;margin-top:2px;}
.ev pre{white-space:pre-wrap;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.8rem;margin:.45rem 0 0;
        color:#334155;background:transparent;border:0;padding:0;}
mark{background:#FFEF9F;padding:0 2px;border-radius:3px;}
.quote{border-left:3px solid #F5C26B;background:#FFFBEF;padding:.35rem .7rem;border-radius:6px;font-size:.85rem;margin-top:.4rem;color:#4B3A12;}
.bar{background:#E8EEF2;border-radius:999px;height:9px;overflow:hidden;}
.bar>div{height:100%;border-radius:999px;}
.notice-safe{background:#FFF8E8;border:1px solid #F2C46D;border-radius:14px;padding:1rem 1.2rem;color:#5A3E0A;}
.notice-disc{background:#F4F7F9;border-radius:10px;padding:.55rem .9rem;font-size:.78rem;color:#64748B;margin-top:.7rem;}
.step{display:flex;gap:.75rem;align-items:flex-start;margin-bottom:.6rem;}
.step .n{flex:0 0 28px;height:28px;border-radius:50%;background:#0E7C86;color:#fff;font-weight:700;text-align:center;line-height:28px;font-size:.85rem;}
.tl{border-left:3px solid #CBD5E1;margin:.5rem 0 .5rem 8px;padding-left:20px;}
.tl-item{position:relative;margin-bottom:14px;background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:.55rem .9rem;}
.tl-item:before{content:"";position:absolute;left:-30px;top:14px;width:12px;height:12px;border-radius:50%;
                background:var(--c,#0E7C86);border:2px solid #fff;box-shadow:0 0 0 2px var(--c,#0E7C86);}
.tl-date{font-size:.76rem;font-weight:700;color:#64748B;} .tl-src{font-size:.72rem;color:#94A3B8;}
.side-status{font-size:.82rem;color:#334155;line-height:1.7;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

NAV = {"🏠  Home": "home", "📤  Upload Report": "upload", "💬  Chat": "chat",
       "🔎  Evidence & Sources": "evidence", "🧪  Laboratory Results": "labs",
       "🕒  Patient Timeline": "timeline", "📊  Evaluation / Admin": "admin", "⚙️  System Information": "system"}
TYPE_COLOR = {"Diagnosis": "#DC2626", "Medication": "#7C5CBF", "Procedure": "#F59E0B",
              "Encounter": "#0E7C86", "Follow-up": "#16A34A", "Laboratory": "#2563EB"}


# ----------------------------------------------------------------------------- helpers
def esc(x):
    return html.escape("" if x is None else str(x))


def pill(text, kind="teal"):
    return f'<span class="pill pill-{kind}">{esc(text)}</span>'


def tile(value, label):
    return f'<div class="tile"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def hero(title, subtitle="", small=False):
    st.markdown(f'<div class="hero {"small" if small else ""}"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>',
                unsafe_allow_html=True)


def card(title, body):
    return f'<div class="card"><h4>{title}</h4><p>{body}</p></div>'


def score_bar(score):
    s = max(0.0, min(1.0, float(score)))
    color = "#16A34A" if s >= 0.6 else "#F59E0B" if s >= 0.3 else "#DC2626"
    return (f'<div style="font-size:.8rem;color:#475569;margin:.5rem 0 3px">Evidence score: <b>{s:.2f}</b> '
            f'<span style="color:#94A3B8">(retrieval relevance, not calibrated confidence)</span></div>'
            f'<div class="bar"><div style="width:{s * 100:.0f}%;background:{color}"></div></div>')


ROUTE_KIND = {"PATIENT_SPECIFIC": "teal", "MEDICAL_KNOWLEDGE": "purple", "HYBRID": "blue",
              "UNSAFE_HIGH_RISK": "red", "UNANSWERABLE": "gray", "NO_RETRIEVAL": "gray"}


def evidence_card(tag, h, question):
    m = h["metadata"]
    patient = m.get("source_type") == "patient_report"
    src = (f'{esc(m.get("document"))} · Page {esc(m.get("page"))} · {esc(m.get("section"))}' if patient
           else f'{esc(m.get("dataset"))} · {esc(m.get("document"))}')
    body = esc(h["text"])
    sup = cit.best_supporting_sentences(question, h["text"], 1)
    if sup:
        s = esc(sup[0][0])
        body = body.replace(s, f"<mark>{s}</mark>", 1)
    st.markdown(
        f'<div class="ev {"" if patient else "k"}">{pill(tag, "teal" if patient else "purple")}'
        f'<span class="src">{src}</span>'
        f'<div class="meta">score {h["evidence_score"]:.3f} · chunk {esc(m.get("chunk_id"))}</div>'
        f'<pre>{body}</pre></div>', unsafe_allow_html=True)


def render_response(r):
    if r.status == "refused_safety":
        st.markdown(f'<div class="notice-safe"><b>🛡️ Safety response</b><br><br>'
                    f'{esc(r.answer).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        return
    if r.status == "error":
        st.error(r.answer)
        return
    pills = pill(r.route or "—", ROUTE_KIND.get(r.route, "gray"))
    pills += pill("Risk: " + r.risk, "green" if r.risk == "INFORMATIONAL" else "red")
    pills += pill(f"{r.latency_s:.1f}s", "gray")
    st.markdown(pills, unsafe_allow_html=True)
    if r.status == "insufficient_evidence":
        st.info("🔍 " + r.answer)
    else:
        st.markdown(r.answer)
    for w in r.warnings:
        st.caption(f"⚠️ {w}")
    if r.evidence_score is not None and r.status == "answered":
        st.markdown(score_bar(r.evidence_score), unsafe_allow_html=True)
    nP, nK, nC = len(r.patient_evidence), len(r.medical_evidence), len(r.citations)
    t_p, t_k, t_c, t_d = st.tabs([f"🧾 Patient evidence ({nP})", f"📚 Medical knowledge ({nK})",
                                  f"🔖 Citations ({nC})", "⚙️ Details"])
    q = r.rewritten_question or r.question
    with t_p:
        if nP:
            for i, h in enumerate(r.patient_evidence, 1):
                evidence_card(f"P{i}", h, q)
        else:
            st.caption("No patient-report evidence was used for this answer.")
    with t_k:
        if nK:
            for i, h in enumerate(r.medical_evidence, 1):
                evidence_card(f"K{i}", h, q)
        else:
            st.caption("No general medical knowledge was used for this answer.")
    with t_c:
        if nC:
            for c in r.citations:
                st.markdown(f'{pill(c["tag"], "teal" if c["source_type"] == "patient_report" else "purple")}'
                            f'<b>{esc(cit.format_source_line(c))}</b>', unsafe_allow_html=True)
                if c["supporting_sentence"]:
                    st.markdown(f'<div class="quote">{esc(c["supporting_sentence"])}</div>', unsafe_allow_html=True)
                st.write("")
        else:
            st.caption("The answer contains no citation tags. The retrieved evidence is shown in the other tabs.")
    with t_d:
        st.json({"route": r.route, "route_reason": r.route_reason, "risk": r.risk,
                 "rewritten_question": r.rewritten_question, "prompt_style": r.prompt_style,
                 "patient_score": r.patient_score, "knowledge_score": r.knowledge_score, "latency_s": r.latency_s})
    st.markdown(f'<div class="notice-disc">🛈 {esc(DISCLAIMER)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- state
@st.cache_resource(show_spinner="Loading models and medical knowledge index...")
def get_pipeline():
    cfg = load_config()
    return cfg, build_pipeline(cfg)


cfg, pipe = get_pipeline()
ss = st.session_state
ss.setdefault("docs", [])
ss.setdefault("chat", [])
ss.setdefault("last", None)
ss.setdefault("pending_q", None)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🩺 MediRAG")
    st.caption("Citation-grounded medical report Q&A")
    choice = st.radio("Navigate", list(NAV), label_visibility="collapsed")
    page = NAV[choice]
    st.divider()
    rep = ss.docs[0].name + (f" (+{len(ss.docs) - 1})" if len(ss.docs) > 1 else "") if ss.docs else "none uploaded"
    kb = f"{len(pipe.medical.store):,} chunks" if pipe.medical else "not built"
    st.markdown(f'<div class="side-status">📄 <b>Report:</b> {esc(rep)}<br>📚 <b>Knowledge base:</b> {kb}<br>'
                f'🤖 <b>LLM:</b> {esc(cfg["llm"]["backend"])}</div>', unsafe_allow_html=True)
    if pipe.medical is None:
        st.warning("Medical index missing. Run `python scripts/build_indexes.py --medical`. Patient-only mode.")
    with st.expander("Advanced"):
        mode = st.selectbox("Pipeline mode", list(PRESETS), index=list(PRESETS).index("medirag"),
                            help="Pick a baseline/ablation configuration to compare behaviour.")
    st.caption("Educational research prototype. Not medical advice.")

# ----------------------------------------------------------------------------- pages
if page == "home":
    hero("MediRAG", "Ask questions about your medical report. Every answer is grounded in your document and cited.")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("🧾 Your report, cited", "Answers point to the exact document, page, section and sentence they come from."), unsafe_allow_html=True)
    c2.markdown(card("📚 Two knowledge sources", "Patient facts come only from your report. General medical context comes from NIH MedQuAD and PubMedQA."), unsafe_allow_html=True)
    c3.markdown(card("🛡️ Safety first", "No diagnoses, no medication changes. Risky questions get a safe, clear redirect."), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Get started")
        for n, t in enumerate(["Open <b>Upload Report</b> and add a PDF, TXT or DOCX.",
                               "Press <b>Process</b> to extract, chunk and index it.",
                               "Go to <b>Chat</b> and ask, e.g. <i>What was my hemoglobin level?</i>",
                               "Check <b>Laboratory Results</b> and <b>Patient Timeline</b> for structured views."], 1):
            st.markdown(f'<div class="step"><div class="n">{n}</div><div>{t}</div></div>', unsafe_allow_html=True)
    with right:
        st.subheader("Pipeline")
        st.markdown("".join(pill(s, "teal") for s in ["Safety check", "Query rewrite", "Router", "Vector + BM25",
                                                       "Reranker", "Evidence gate", "LLM", "Citation check"]),
                    unsafe_allow_html=True)
        st.caption("Demo reports are synthetic (Synthea-style).")
    st.markdown(f'<div class="notice-safe">⚠️ {esc(DISCLAIMER)}</div>', unsafe_allow_html=True)

elif page == "upload":
    hero("Upload & process", "Add one or more reports. They are indexed locally on this machine.", small=True)
    files = st.file_uploader("Drop PDF, TXT or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if files and st.button("⚙️ Process reports", type="primary"):
        up = Path(cfg["paths"]["uploads_dir"])
        up.mkdir(parents=True, exist_ok=True)
        docs, chunks = [], []
        with st.status("Processing...", expanded=True) as status:
            for f in files:
                p = up / f.name
                p.write_bytes(f.getbuffer())
                st.write(f"📄 Extracting text: **{f.name}**")
                d = process_document(p, cfg["chunking"]["chunk_tokens"], cfg["chunking"]["overlap_tokens"])
                st.write(f"→ {len(d.pages)} page(s) · {len(d.blocks)} section block(s) · {len(d.chunks)} chunk(s) · {d.doc_type}")
                for w in d.warnings:
                    st.warning(w)
                docs.append(d)
                chunks += d.chunks
            st.write("🧮 Embedding and indexing (FAISS + BM25)...")
            t0 = time.time()
            pipe.index_patient_chunks(chunks, save_dir=cfg["paths"]["patient_store"])
            st.write(f"→ indexed {len(chunks)} chunks in {time.time() - t0:.1f}s")
            status.update(label="Processing complete", state="complete")
        ss.docs, ss.chat, ss.last = docs, [], None
        st.success("Report ready. Open **Chat** to start asking questions.")
    if ss.docs:
        n_labs = sum(len(d.structured["lab_results"]) for d in ss.docs)
        cols = st.columns(4)
        for col, (v, l) in zip(cols, [(len(ss.docs), "Documents"), (sum(len(d.pages) for d in ss.docs), "Pages"),
                                      (sum(len(d.chunks) for d in ss.docs), "Chunks"), (n_labs, "Lab results found")]):
            col.markdown(tile(v, l), unsafe_allow_html=True)
        st.write("")
        for d in ss.docs:
            with st.expander(f"📄 {d.name}", expanded=len(ss.docs) == 1):
                s = d.structured
                a, b = st.columns(2)
                with a:
                    st.markdown("**Patient information**")
                    info = s["patient_information"]
                    if info:
                        st.dataframe(pd.DataFrame(list(info.items()), columns=["Field", "Value"]), hide_index=True)
                    else:
                        st.caption("None detected.")
                    st.markdown("**Diagnoses**")
                    st.write("\n".join(f"- {x['text']}" for x in s["diagnoses"]) or "None detected.")
                with b:
                    st.markdown("**Medications**")
                    st.write("\n".join(f"- {x['text']}" for x in s["medications"]) or "None detected.")
                    st.markdown("**Sections found**")
                    st.markdown("".join(pill(x, "gray") for x in dict.fromkeys(bk["section"] for bk in d.blocks)),
                                unsafe_allow_html=True)
    elif not files:
        st.info("No report processed yet. Upload a file above to begin.")

elif page == "chat":
    hero("Ask about your report", "Answers are grounded in your document and general medical knowledge, with citations.", small=True)
    top = st.columns([6, 1])
    if top[1].button("🗑️ Clear", help="Clear the conversation"):
        ss.chat, ss.last = [], None
        st.rerun()
    if not ss.docs:
        st.info("No report uploaded: only general medical-knowledge questions can be answered.")

    q = ss.pending_q
    ss.pending_q = None
    if not ss.chat and not q:
        st.markdown("**Try asking:**")
        if ss.docs and ss.docs[0].structured["lab_results"]:
            first_test = ss.docs[0].structured["lab_results"][0]["test"]
            sugg = [f"What was my {first_test} result?", "Which medications are listed in my report?",
                    f"Is my {first_test} within the reference range in the report?", "What is anemia?",
                    "Should I stop taking my medicine?"]
        else:
            sugg = ["What is anemia?", "What causes type 2 diabetes?", "What are the symptoms of hypertension?"]
        cols = st.columns(2)
        for i, s in enumerate(sugg):
            if cols[i % 2].button(s, key=f"sugg{i}"):
                ss.pending_q = s
                st.rerun()

    for m in ss.chat:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🩺"):
            if m["role"] == "assistant" and m.get("resp") is not None:
                render_response(m["resp"])
            else:
                st.markdown(m["content"])

    typed = st.chat_input("Ask a question, e.g. What was my hemoglobin level?")
    q = typed or q
    if q:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(q)
        hist = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Retrieving evidence and generating the answer..."):
                r = pipe.answer(q, history=hist, options=PRESETS[mode],
                                doc_filter=ss.docs[0].name if len(ss.docs) == 1 else None)
            render_response(r)
        ss.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer, "resp": r}]
        ss.last = r

elif page == "evidence":
    hero("Evidence & sources", "Everything the last answer was based on.", small=True)
    r = ss.last
    if r is None:
        st.info("💬 Nothing here yet. Ask a question in **Chat** and the retrieved evidence will appear on this page.")
    else:
        st.markdown(f"**Question:** {esc(r.question)}", unsafe_allow_html=True)
        st.markdown(pill(r.route, ROUTE_KIND.get(r.route, "gray")) + pill("Status: " + r.status, "gray"),
                    unsafe_allow_html=True)
        q = r.rewritten_question or r.question
        for i, h in enumerate(r.patient_evidence, 1):
            evidence_card(f"P{i}", h, q)
        for i, h in enumerate(r.medical_evidence, 1):
            evidence_card(f"K{i}", h, q)
        if not (r.patient_evidence or r.medical_evidence):
            st.caption("No evidence was retrieved for this question.")
    if ss.docs:
        st.subheader("Indexed documents")
        for d in ss.docs:
            with st.expander(d.name):
                st.dataframe(pd.DataFrame([{"Page": b["page"], "Section": b["section"], "Words": len(b["text"].split())}
                                           for b in d.blocks]), hide_index=True)

elif page == "labs":
    hero("Laboratory results", "Values extracted from your report. Colours use only ranges printed in the report.", small=True)
    labs = [l for d in ss.docs for l in d.structured["lab_results"]]
    if not labs:
        st.info("🧪 No laboratory results yet. Upload and process a report first.")
    else:
        df = labs_to_dataframe(labs)
        counts = df["Status"].value_counts()
        cols = st.columns(5)
        for col, (v, l) in zip(cols, [(len(df), "Total tests"),
                                      (int(counts.get("Within report range", 0)), "Within range"),
                                      (int(counts.get("Below report range", 0)), "Below range"),
                                      (int(counts.get("Above report range", 0)), "Above range"),
                                      (int(counts.get("No reference range in report", 0)), "No range given")]):
            col.markdown(tile(v, l), unsafe_allow_html=True)
        st.write("")
        f1, f2 = st.columns(2)
        sel_status = f1.multiselect("Filter by status", sorted(df["Status"].unique()))
        sel_test = f2.multiselect("Filter by test", sorted(df["Test"].unique()))
        view = df
        if sel_status:
            view = view[view["Status"].isin(sel_status)]
        if sel_test:
            view = view[view["Test"].isin(sel_test)]

        def color(row):
            s = row["Status"]
            c = ("background-color:#FDE8E8" if ("Above" in s or "Below" in s)
                 else "background-color:#E3F6E8" if "Within" in s else "")
            return [c] * len(row)

        st.dataframe(view.style.apply(color, axis=1), hide_index=True)
        st.caption("Colours appear only when the report itself prints a reference range. This is not a diagnosis; "
                   "where no range is printed, none is assumed.")

elif page == "timeline":
    hero("Patient timeline", "Dated events and lab values, in order.", small=True)
    if not ss.docs:
        st.info("🕒 Upload a report first.")
    else:
        tl = build_timeline([d.structured for d in ss.docs])
        if tl.empty:
            st.info("No dated events were found in this report.")
        else:
            types = sorted(tl["Type"].unique())
            show = st.multiselect("Event types", types, default=types)
            items = ""
            for r_ in tl[tl["Type"].isin(show)].itertuples():
                items += (f'<div class="tl-item" style="--c:{TYPE_COLOR.get(r_.Type, "#0E7C86")}">'
                          f'<div class="tl-date">{esc(r_.Date)} · {esc(r_.Type)}</div><div>{esc(r_.Event)}</div>'
                          f'<div class="tl-src">{esc(r_.Source)}</div></div>')
            st.markdown(f'<div class="tl">{items}</div>', unsafe_allow_html=True)
        labs = [l for d in ss.docs for l in d.structured["lab_results"]]
        tests = sorted({l["test"] for l in labs})
        if tests:
            st.subheader("Lab trend")
            t = st.selectbox("Test", tests)
            pts, text = lab_trend(labs, t)
            if len(pts) >= 2:
                st.line_chart(pd.DataFrame({"date": [p["date"] for p in pts], t: [p["value"] for p in pts]}).set_index("date"))
            st.info(text)

elif page == "admin":
    hero("Evaluation & admin", "Experiment results and tools.", small=True)
    t_res, t_router, t_cfg = st.tabs(["📈 Results", "🧭 Router tester", "🛠️ Configuration"])
    with t_res:
        res = Path(cfg["paths"]["results_dir"])
        files = sorted(res.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            st.info("No results yet. Run `python -m evaluation.experiments --exp ablation` (see README).")
        else:
            f = st.selectbox("Result file", files, format_func=lambda p: p.name)
            st.dataframe(pd.read_csv(f))
    with t_router:
        tq = st.text_input("Type a question to see how it is classified")
        if tq:
            from retrieval.query_router import route_query
            from safety.risk_classifier import classify_risk
            risk, d = classify_risk(tq), route_query(tq, bool(ss.docs))
            st.markdown(pill(d.route.value, ROUTE_KIND.get(d.route.value, "gray")) +
                        pill("Risk: " + risk.level.value, "green" if risk.level.value == "INFORMATIONAL" else "red"),
                        unsafe_allow_html=True)
            st.caption(f"Router reason: {d.reason} · risk category: {risk.category}")
    with t_cfg:
        st.json({k: cfg[k] for k in ("embedding", "chunking", "retrieval", "reranker", "llm")})

elif page == "system":
    hero("System information", "What is loaded right now.", small=True)
    cols = st.columns(4)
    for col, (v, l) in zip(cols, [(f"{len(pipe.medical.store):,}" if pipe.medical else "0", "Knowledge chunks"),
                                  (f"{len(pipe.patient.store):,}" if pipe.patient else "0", "Patient chunks"),
                                  (cfg["llm"]["backend"], "LLM backend"),
                                  ("on" if pipe.reranker else "off", "Reranker")]):
        col.markdown(tile(v, l), unsafe_allow_html=True)
    st.write("")
    st.json({"embedding_model": cfg["embedding"]["model_name"], "reranker_model": cfg["reranker"]["model_name"],
             "llm_model": cfg["llm"]["model_name"], "top_k": cfg["retrieval"]["top_k_final"],
             "chunk_tokens": cfg["chunking"]["chunk_tokens"]})