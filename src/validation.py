"""
Grounding check: verifies that generated quiz questions actually connect to
the retrieved context, rather than just trusting the LLM's promise not to
hallucinate.

Prompt instructions like "don't use facts outside the context" are a soft
guarantee -- the model can still ignore them. This module adds a second,
independent check after generation. It's a simple word-overlap heuristic
(not another LLM call) so it's fast and has zero extra API cost, which
matters if this runs on every quiz generation.
"""

import re

from src.config import GROUNDING_OVERLAP_THRESHOLD

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "from", "has",
    "have", "had", "not", "no", "which", "who", "what", "when", "where",
}


def _tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _overlap_ratio(explanation, context):
    exp_tokens = _tokenize(explanation)
    ctx_tokens = _tokenize(context)
    if not exp_tokens:
        return 0.0
    matched = exp_tokens & ctx_tokens
    return len(matched) / len(exp_tokens)


def check_grounding(questions, context):
    """
    Annotates each question dict with:
      - "grounding_score": float 0-1, share of the explanation's meaningful
        words that also appear somewhere in the retrieved context
      - "grounded": bool, whether that score clears the configured threshold

    This doesn't prove the question is factually correct -- it's a proxy
    signal that the explanation is actually tied to retrieved text rather
    than invented from the model's training data. Flag ungrounded
    questions in the UI so a human reviewer (or a regeneration step) can
    catch them before they go out.
    """
    annotated = []
    for q in questions:
        explanation = q.get("explanation", "")
        score = _overlap_ratio(explanation, context)
        q_copy = dict(q)
        q_copy["grounding_score"] = round(score, 2)
        q_copy["grounded"] = score >= GROUNDING_OVERLAP_THRESHOLD
        annotated.append(q_copy)
    return annotated
