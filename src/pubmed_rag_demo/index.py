from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from .corpus import load_txt_corpus
from .retriever import BM25Retriever

def build_bm25_from_dir(dir_path: str | Path) -> BM25Retriever:
    docs, ids = load_txt_corpus(dir_path)
    r = BM25Retriever()
    r.add(docs, ids)
    return r

def topk_ids_scores(r: BM25Retriever, query: str, k: int = 3) -> List[Tuple[str, float]]:
    return r.query(query, k=k)
