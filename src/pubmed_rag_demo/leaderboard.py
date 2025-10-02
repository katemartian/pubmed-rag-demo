from __future__ import annotations
from pathlib import Path
import json
import typer
from .index import build_bm25_from_dir
from .eval import context_hit_rate, retrieval_precision_at_k

app = typer.Typer(add_completion=False)

@app.command()
def main(
    data_dir: str = typer.Argument(..., help="Folder with *.txt abstracts"),
    qa_path: str = typer.Argument(..., help="JSONL with QA pairs"),
    max_k: int = typer.Option(3, "--max-k", help="Test values of k from 1..max_k"),
    out_path: str = typer.Option("leaderboard.md", "--out", help="Markdown table output"),
):
    # load QA
    qa_pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qa_pairs.append((obj["question"], obj["answer"]))

    retriever = build_bm25_from_dir(data_dir)

    rows = []
    for k in range(1, max_k + 1):
        hit = context_hit_rate(qa_pairs, retriever, k=k)
        prec = retrieval_precision_at_k(qa_pairs, retriever, k=k)
        rows.append({"k": k, **hit, **prec})

    # write Markdown table
    lines = ["| k | hit_rate | precision@k |", "|---|----------|-------------|"]
    for row in rows:
        lines.append(f"| {row['k']} | {row['hit_rate']:.3f} | {row['precision_at_k']:.3f} |")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"Saved leaderboard to {out_path}")

if __name__ == "__main__":
    app()
