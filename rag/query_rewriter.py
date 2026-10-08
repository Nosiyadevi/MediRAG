"""Resolve follow-up questions ("Is that normal?") using chat history. Rewrite stays internal."""
import re

PRONOUN_RE = re.compile(r"\b(it|that|this|those|these|they|them|its|the same)\b", re.I)


def needs_rewrite(question: str, history: list) -> bool:
    return bool(history) and len(question.split()) <= 12 and bool(PRONOUN_RE.search(question))


def rewrite_rule_based(question: str, history: list) -> str:
    prev_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if not prev_q:
        return question
    return f"{question.strip()} (referring to the previous question: {prev_q.strip()})"


def rewrite_with_llm(question: str, history: list, llm) -> str:
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    system = ("Rewrite the user's last question as a standalone question, resolving pronouns using the "
              "conversation. Output only the rewritten question. Do not answer it.")
    try:
        out = llm.generate(system, f"Conversation:\n{convo}\n\nLast question: {question}").strip()
        return out.splitlines()[0] if out else question
    except Exception:
        return rewrite_rule_based(question, history)


def rewrite_query(question, history, llm=None, use_llm=False):
    if not needs_rewrite(question, history):
        return question
    if use_llm and llm is not None and llm.backend != "extractive":
        return rewrite_with_llm(question, history, llm)
    return rewrite_rule_based(question, history)
