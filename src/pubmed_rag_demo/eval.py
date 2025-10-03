from __future__ import annotations
from typing import List, Dict, Tuple, Set
import re

def context_hit_rate(
    qa_pairs: List[Tuple[str, str]],  # (question, expected_answer_substring)
    retriever,
    k: int = 3,
) -> Dict[str, float]:
    """
    For each QA pair, check if any of the top-k retrieved docs
    contains the expected answer substring (case-insensitive).
    Returns {"hit_rate": float}
    """
    hits = 0
    for q, answer in qa_pairs:
        results = retriever.query(q, k=k)
        if not results:
            continue
        for doc_id, _ in results:
            # we don't store docs inside retriever here, so assume retriever has docs list
            idx = retriever._doc_ids.index(doc_id)
            doc_text = retriever._docs[idx]
            if answer.lower() in doc_text.lower():
                hits += 1
                break
    total = len(qa_pairs)
    return {"hit_rate": hits / total if total > 0 else 0.0}

def retrieval_precision_at_k(
    qa_pairs: List[Tuple[str, str]],
    retriever,
    k: int = 3,
) -> Dict[str, float]:
    """
    For each QA pair, count how many of the top-k retrieved docs contain the expected
    answer substring (case-insensitive). Average that fraction across all QA pairs.
    Returns {"precision_at_k": float}
    """

    if k <= 0:
        return {"precision_at_k": 0.0}

    precisions: List[float] = []
    for q, answer in qa_pairs:
        results = retriever.query(q, k=k)
        if not results:
            precisions.append(0.0)
            continue
        hits = 0
        for doc_id, _ in results:
            idx = retriever._doc_ids.index(doc_id)
            doc_text = retriever._docs[idx]
            if answer.lower() in doc_text.lower():
                hits += 1
        precisions.append(hits / k)
    # average over all QA pairs
    return {"precision_at_k": sum(precisions) / len(precisions) if precisions else 0.0}

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS: Set[str] = {
    "the","a","an","and","or","of","to","in","on","for","with","by","is","are","was","were",
    "that","this","it","as","at","from","into","can","be","about","which"
}

def _tokens(text: str) -> Set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS}

def faithfulness_overlap(
    qa_pairs: List[Tuple[str, str]],
    candidates: List[str],
    retriever,
    k: int = 3,
    threshold: float = 0.6,
) -> Dict[str, float]:
    """
    For each (Q,A*) + candidate answer C:
      - retrieve top-k docs for Q
      - compute Jaccard-like overlap: |tokens(C) ∩ tokens(doc)| / |tokens(C)|
      - if any doc reaches >= threshold, count as faithful
    Returns {"faithfulness": mean_faithful_rate}
    """
    assert len(qa_pairs) == len(candidates), "qa_pairs and candidates must align"
    faithful = 0
    total = len(qa_pairs)
    for (q, _), cand in zip(qa_pairs, candidates):
        results = retriever.query(q, k=k)
        cand_toks = _tokens(cand)
        if not cand_toks or not results:
            continue
        ok = False
        for doc_id, _ in results:
            idx = retriever._doc_ids.index(doc_id)
            doc_toks = _tokens(retriever._docs[idx])
            overlap = len(cand_toks & doc_toks) / max(1, len(cand_toks))
            if overlap >= threshold:
                ok = True
                break
        if ok:
            faithful += 1
    return {"faithfulness": (faithful / total) if total > 0 else 0.0}
