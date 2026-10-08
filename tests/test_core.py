import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random

from extraction.lab_extractor import parse_lab_line, derive_status
from ingestion.chunker import chunk_text
from ingestion.document_processor import process_document
from retrieval.query_router import Route, route_query
from safety.guardrails import post_check
from safety.risk_classifier import Risk, classify_risk
from scripts.generate_reports import build_builtin, write_pdf


def test_chunker_keeps_rows_whole():
    text = "\n".join(f"Test{i} | {i}.0 g/dL | 1.0 - 9.0 | " for i in range(200))
    chunks = chunk_text(text, 64, 8)
    assert len(chunks) > 1
    for c in chunks:
        for line in c.split("\n"):
            assert line.startswith("Test") and line.count("|") == 3


def test_lab_parsing_and_status():
    r = parse_lab_line("Hemoglobin | 10.2 g/dL | 12.0 - 15.5 | L", True)
    assert r[:3] == ("Hemoglobin", 10.2, "g/dL") and r[3] == "12.0 - 15.5"
    assert derive_status(10.2, r[3], r[4]) == "Below report range"
    assert derive_status(5.0, None, None) == "No reference range in report"
    assert parse_lab_line("Lisinopril 10 mg once daily", False) is None
    g = parse_lab_line("Glucose: 142 mg/dL (70-99)", True)
    assert g[1] == 142.0 and g[3] == "70-99"


def test_router_examples():
    assert route_query("What was my hemoglobin level?").route == Route.PATIENT
    assert route_query("What is anemia?").route == Route.KNOWLEDGE
    assert route_query("Does my report contain signs associated with anemia?").route == Route.HYBRID
    assert route_query("What was my hemoglobin level?", has_patient_report=False).route == Route.UNANSWERABLE


def test_risk_classifier():
    assert classify_risk("Should I stop taking my blood pressure medicine?").level == Risk.HIGH_RISK
    assert classify_risk("I have chest pain").level == Risk.HIGH_RISK
    assert classify_risk("Do I have diabetes?").level == Risk.CLINICAL_DECISION
    assert classify_risk("When did the patient start metformin?").level == Risk.INFORMATIONAL
    assert classify_risk("What is metformin used for?").level == Risk.INFORMATIONAL


def test_post_check_blocks_directive_advice():
    ans, flags = post_check("You should stop taking your medication today.")
    assert flags and "can't safely" in ans


def test_end_to_end_with_extractive(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIRAG_TEST_MODE", "1")
    from config import load_config
    from rag.chains import build_pipeline
    pid, text = build_builtin(1, random.Random(1))
    pdf = tmp_path / "SYN-0001.pdf"
    write_pdf(text, pdf)
    cfg = load_config()
    doc = process_document(pdf, 256, 32)
    assert doc.structured["lab_results"] and doc.structured["medications"]
    pipe = build_pipeline(cfg, load_medical=False)
    pipe.index_patient_chunks(doc.chunks)
    lab = doc.structured["lab_results"][0]
    r = pipe.answer(f"What was my {lab['test']} result?")
    assert r.status == "answered" and r.route == "PATIENT_SPECIFIC"
    assert str(lab["value"]) in r.answer or f"{lab['value']:g}" in r.answer
    assert r.citations and r.citations[0]["page"] == lab["source_page"]
    assert pipe.answer("Should I stop taking my medicine?").status == "refused_safety"
    assert pipe.answer("What happened to the patient in 1987?").status == "insufficient_evidence"
