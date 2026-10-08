"""Deterministic query router (rules). Evaluated for accuracy on the benchmark.

UNANSWERABLE is mostly decided AFTER retrieval (evidence gate); here it is returned only when a
patient-specific question is asked but no report has been uploaded.
"""
import re
from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    PATIENT = "PATIENT_SPECIFIC"
    KNOWLEDGE = "MEDICAL_KNOWLEDGE"
    HYBRID = "HYBRID"
    UNSAFE = "UNSAFE_HIGH_RISK"
    UNANSWERABLE = "UNANSWERABLE"


@dataclass
class RouteDecision:
    route: Route
    reason: str


PATIENT_CUES = re.compile(
    r"\b(my|mine|me|i|patient'?s?|the patient|this report|the report|my report|uploaded|discharge[d]?|admitted|"
    r"prescribed|according to the report|in the report)\b|\b(was|were)\b.{0,40}\b(level|value|result|reading|count|score)s?\b", re.I)
KNOWLEDGE_STRONG = re.compile(
    r"\b(define|definition of|symptoms? of|causes? of|risk factors?|how is .{0,40}(diagnosed|treated)|"
    r"difference between|what does .{0,40} mean|normal range|reference range|explain|treatment (for|of)|"
    r"what causes|why do(es)?)\b", re.I)
KNOWLEDGE_WEAK = re.compile(r"^\s*(what|who)\s+(is|are)\b", re.I)
HYBRID_CUES = re.compile(
    r"\b(signs?|indicat\w*|suggest\w*|associated with|consistent with|relat(?:ed|es) to|mean|means|normal|abnormal|"
    r"concern\w*|risk|significan\w*|high|low)\b", re.I)


def route_query(question: str, has_patient_report: bool = True) -> RouteDecision:
    p = bool(PATIENT_CUES.search(question))
    ks = bool(KNOWLEDGE_STRONG.search(question))
    kw = bool(KNOWLEDGE_WEAK.search(question))
    h = bool(HYBRID_CUES.search(question))
    if not has_patient_report:
        if p and not ks:
            return RouteDecision(Route.UNANSWERABLE, "patient-specific question but no report uploaded")
        return RouteDecision(Route.KNOWLEDGE, "no report uploaded; using medical knowledge only")
    if p and (ks or h):
        return RouteDecision(Route.HYBRID, "patient cue + knowledge/interpretation cue")
    if p:
        return RouteDecision(Route.PATIENT, "patient cue")
    if ks or kw:
        return RouteDecision(Route.KNOWLEDGE, "general medical knowledge cue")
    return RouteDecision(Route.HYBRID, "ambiguous; searching both sources")
