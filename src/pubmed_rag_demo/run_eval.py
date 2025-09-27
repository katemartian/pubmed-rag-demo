from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import typer
from .index import build_bm25_from_dir
from .eval import context_hit_rate

app = typer.Typer(add_completion=False)

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

    # compute metrics
    metrics = context_hit_rate(qa_pairs, retriever, k=k)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = out_p / f"metrics_{ts}.json"
    report_path = out_p / f"report_{ts}.md"

    # save metrics.json
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"k": k, **metrics}, f, ensure_ascii=False, indent=2)

    # save markdown report
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- `k`: **{k}**\n")
        f.write(f"- `hit_rate`: **{metrics['hit_rate']:.3f}**\n")

    typer.echo(f"Saved: {metrics_path}")
    typer.echo(f"Saved: {report_path}")

if __name__ == "__main__":
    app()
