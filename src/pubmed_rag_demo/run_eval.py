from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
import typer
from .index import build_bm25_from_dir
from .eval import context_hit_rate, retrieval_precision_at_k, faithfulness_overlap

app = typer.Typer(add_completion=False)

_WORD = re.compile(r"[a-z0-9]+")

def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))

def naive_candidates(qa_pairs, retriever, k_for_candidate: int = 1) -> list[str]:
    """
    For each question:
      - retrieve top-1 doc (or top-k and use the first)
      - split into sentences
      - pick the sentence with maximum token overlap with the question tokens
    Returns a list of candidate answers (strings) aligned with qa_pairs.
    """
    candidates: list[str] = []
    for q, _ in qa_pairs:
        results = retriever.query(q, k=k_for_candidate)
        if not results:
            candidates.append("")
            continue
        # take the top doc
        doc_id, _ = results[0]
        idx = retriever._doc_ids.index(doc_id)
        doc_text = retriever._docs[idx]

        # split into crude sentences
        sentences = re.split(r"(?<=[.!?])\s+", doc_text.strip())
        q_toks = _tokens(q)

        best_sent = ""
        best_score = -1.0
        for s in sentences:
            s_toks = _tokens(s)
            score = len(q_toks & s_toks)
            if score > best_score:
                best_score = score
                best_sent = s
        candidates.append(best_sent)
    return candidates

@app.command()
def main(
    data_dir: str = typer.Argument(..., help="Folder with *.txt abstracts"),
    qa_path: str = typer.Argument(..., help="JSONL with fields: question, answer"),
    k: int = typer.Option(3, "--k", "-k", help="Top-k to retrieve"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Where to save metrics/report"),
):
    data_p = Path(data_dir)
    qa_p = Path(qa_path)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # load QA
    qa_pairs = []
    with qa_p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qa_pairs.append((obj["question"], obj["answer"]))

    # build retriever
    retriever = build_bm25_from_dir(data_p)

    # retrieval metrics
    hit = context_hit_rate(qa_pairs, retriever, k=k)
    prec = retrieval_precision_at_k(qa_pairs, retriever, k=k)

    # naive candidates + faithfulness
    cand = naive_candidates(qa_pairs, retriever, k_for_candidate=1)
    faith = faithfulness_overlap(qa_pairs, cand, retriever, k=k, threshold=0.6)

    # collate + save
    metrics = {"k": k, **hit, **prec, **faith}
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = out_p / f"metrics_{ts}.json"
    report_path = out_p / f"report_{ts}.md"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- `k`: **{k}**\n")
        f.write(f"- `hit_rate`: **{metrics['hit_rate']:.3f}**\n")
        f.write(f"- `precision_at_k`: **{metrics['precision_at_k']:.3f}**\n")
        f.write(f"- `faithfulness`: **{metrics['faithfulness']:.3f}**\n")

    typer.echo(f"Saved: {metrics_path}")
    typer.echo(f"Saved: {report_path}")

if __name__ == "__main__":
    app()
