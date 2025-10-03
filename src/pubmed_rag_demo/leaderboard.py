from __future__ import annotations
from pathlib import Path
import json
import re
import typer
from .index import build_bm25_from_dir
from .eval import context_hit_rate, retrieval_precision_at_k, faithfulness_overlap

app = typer.Typer(add_completion=False)

_WORD = re.compile(r"[a-z0-9]+")

def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))

def load_qa(qa_path: str) -> list[tuple[str, str]]:
    qa_pairs: list[tuple[str, str]] = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qa_pairs.append((obj["question"], obj["answer"]))
    return qa_pairs

def naive_candidates(qa_pairs, retriever, k_for_candidate: int = 1) -> list[str]:
    cands: list[str] = []
    for q, _ in qa_pairs:
        results = retriever.query(q, k=k_for_candidate)
        if not results:
            cands.append("")
            continue
        doc_id, _ = results[0]
        idx = retriever._doc_ids.index(doc_id)
        doc_text = retriever._docs[idx]
        sentences = re.split(r"(?<=[.!?])\s+", doc_text.strip())
        q_toks = _tokens(q)
        best_sent, best_score = "", -1.0
        for s in sentences:
            s_toks = _tokens(s)
            score = len(q_toks & s_toks)
            if score > best_score:
                best_score, best_sent = score, s
        cands.append(best_sent)
    return cands

@app.command()
def main(
    data_dir: str = typer.Argument(..., help="Folder with *.txt abstracts"),
    qa_path: str = typer.Argument(..., help="JSONL with QA pairs"),
    max_k: int = typer.Option(3, "--max-k", help="Test values of k from 1..max_k"),
    out_path: str = typer.Option("leaderboard.md", "--out", help="Markdown table output"),
):
    qa_pairs = load_qa(qa_path)
    retriever = build_bm25_from_dir(data_dir)

    rows = []
    for k in range(1, max_k + 1):
        hit = context_hit_rate(qa_pairs, retriever, k=k)
        prec = retrieval_precision_at_k(qa_pairs, retriever, k=k)
        cands = naive_candidates(qa_pairs, retriever, k_for_candidate=1)
        faith = faithfulness_overlap(qa_pairs, cands, retriever, k=k, threshold=0.6)
        rows.append({"k": k, **hit, **prec, **faith})

    lines = ["| k | hit_rate | precision@k | faithfulness |", "|---|----------|-------------|--------------|"]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['hit_rate']:.3f} | {row['precision_at_k']:.3f} | {row['faithfulness']:.3f} |"
        )

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"Saved leaderboard to {out_path}")

if __name__ == "__main__":
    app()
