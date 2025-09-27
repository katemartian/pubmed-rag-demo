from __future__ import annotations
from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> List[str]:
    # lowercase and keep only alphanumerics (splits on punctuation)
    return _TOKEN_RE.findall(text.lower())

class BM25Retriever:
    def __init__(self) -> None:
        self._docs: List[str] = []
        self._doc_ids: List[str] = []
        self._bm25: BM25Okapi | None = None

    def add(self, docs: List[str], ids: List[str] | None = None) -> None:
        if ids is None:
            ids = [str(i) for i in range(len(self._docs), len(self._docs) + len(docs))]
        if len(ids) != len(docs):
            raise ValueError("ids and docs must have same length")
        self._docs.extend(docs)
        self._doc_ids.extend(ids)
        tokenized = [_tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        # return top-k even if scores are 0.0 (caller can decide how to use)
        return [(self._doc_ids[i], float(scores[i])) for i in ranked_idx]
