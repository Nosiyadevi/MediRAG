"""Safety responses, disclaimer and post-generation output checks."""
import re

DISCLAIMER = ("MediRAG is an educational research prototype, not a medical device. It does not diagnose, "
              "prescribe, or replace a qualified healthcare professional. Please discuss your results and any "
              "treatment decisions with your doctor.")
INSUFFICIENT = "I could not find sufficient evidence in the available documents to answer this question."

_RESPONSES = {
    "emergency": ("This may be an emergency. I cannot assess urgent symptoms. Please contact your local emergency "
                  "number right away (for example 112 in India, 911 in the US) or go to the nearest emergency "
                  "department. If you are having thoughts of harming yourself, please reach out to a crisis "
                  "helpline or someone you trust immediately."),
    "medication_change": ("I can't safely advise you to start, stop, skip, or change a medication or its dose. "
                          "That decision depends on your full medical history and must be made with the doctor "
                          "who prescribed it (or your pharmacist). I can show you what your report says about "
                          "your medications or explain general information about a drug if you ask."),
    "diagnosis_or_treatment": ("I can't diagnose conditions or decide on treatment. I can point out what your "
                               "report documents and explain general medical terms, but a diagnosis requires a "
                               "clinician who can examine you and consider your full history."),
}


def safety_response(category: str) -> str:
    return _RESPONSES.get(category, _RESPONSES["diagnosis_or_treatment"]) + "\n\n" + DISCLAIMER


_UNSAFE_OUTPUT = [
    r"\byou\s+(?:should|must|need\s+to|have\s+to)\s+(?:stop|start|increase|decrease|double|skip|discontinue|quit)\b[^.\n]{0,50}"
    r"\b(?:medication|medicine|dose|pill|tablet|drug|insulin|statin)\b",
    r"\b(?:i|we)\s+(?:recommend|advise|suggest)\s+(?:that\s+)?you\s+(?:stop|start|increase|decrease|skip)\b",
    r"\byou\s+(?:definitely|certainly|clearly|undoubtedly)\s+(?:have|are\s+suffering)\b",
]


def post_check(answer: str):
    """Return (answer, flags). Replaces answers that give directive medication advice or definitive diagnoses."""
    flags = [p for p in _UNSAFE_OUTPUT if re.search(p, answer, re.I)]
    if flags:
        return safety_response("medication_change"), ["unsafe_output_replaced"]
    return answer, []
