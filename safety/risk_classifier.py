"""Pre-generation risk classifier (deterministic rules).

INFORMATIONAL      -> answer normally
CLINICAL_DECISION  -> diagnosis / treatment decision requests: safe redirect
HIGH_RISK          -> emergencies and medication start/stop/change: safe redirect
"""
import re
from dataclasses import dataclass
from enum import Enum


class Risk(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    CLINICAL_DECISION = "CLINICAL_DECISION"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class RiskResult:
    level: Risk
    category: str = "none"
    matched: str = ""


_ACT = r"(?:stop|quit|skip|discontinue|increase|decrease|double|halve|reduce|change|switch|restart|start|miss|cut)"
_MED = r"(?:medication|medicine|medicines|meds|tablet|tablets|pill|pills|dose|dosage|drug|insulin|statin|antibiotic|prescription|inhaler)"
EMERGENCY = [r"\bchest pain\b", r"\b(can'?t|cannot|can not|unable to) breathe\b", r"\bdifficulty breathing\b",
             r"\bsevere bleeding\b", r"\bsuicid", r"\bkill myself\b", r"\boverdos", r"\bunconscious\b",
             r"\bheart attack\b", r"\bstroke\b.*\b(now|symptoms)\b", r"\bpassing out\b"]
MED_CHANGE = [
    rf"\b(?:should|can|could|may|shall|must|do)\s+(?:i|we)\b.{{0,30}}\b{_ACT}\b",
    rf"\bis\s+it\s+(?:ok|okay|safe|fine|alright)\s+to\s+{_ACT}\b",
    rf"\b(?:want|going|plan(?:ning)?|trying|need)\s+to\s+{_ACT}\b.{{0,40}}\b{_MED}\b",
    rf"\bhow\s+(?:much|many|often)\b.{{0,40}}\bshould\s+i\s+(?:take|use)\b",
    rf"\b{_ACT}\s+(?:taking|using)\s+(?:my|the)\b",
]
DIAGNOSIS = [r"\bdo\s+i\s+have\b", r"\bam\s+i\s+(?:suffering|diabetic|anemic|anaemic|sick|dying|going to die)\b",
             r"\bdiagnose\s+(?:me|my)\b", r"\bwhat\s+(?:disease|illness|condition)\s+do\s+i\s+have\b",
             r"\bdo\s+i\s+need\s+(?:surgery|treatment|an?\s+operation|chemotherapy)\b",
             r"\bwhat\s+treatment\s+should\b", r"\bwhat\s+should\s+i\s+take\s+for\b"]


def classify_risk(question: str) -> RiskResult:
    q = question.lower()
    for pats, level, cat in ((EMERGENCY, Risk.HIGH_RISK, "emergency"),
                             (MED_CHANGE, Risk.HIGH_RISK, "medication_change"),
                             (DIAGNOSIS, Risk.CLINICAL_DECISION, "diagnosis_or_treatment")):
        for p in pats:
            m = re.search(p, q)
            if m:
                return RiskResult(level, cat, m.group(0))
    return RiskResult(Risk.INFORMATIONAL)
