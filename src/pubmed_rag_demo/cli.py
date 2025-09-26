from __future__ import annotations
import json
from pathlib import Path
import typer
from .index import build_bm25_from_dir, topk_ids_scores

app = typer.Typer(add_completion=False)

@app.command()
def query(
    data_dir: str = typer.Argument(..., help="Folder with *.txt abstracts"),
    q: str = typer.Option(..., "--q", "-q", help="Query text"),
    k: int = typer.Option(3, "--k", "-k", help="Top-k results"),
):
    r = build_bm25_from_dir(Path(data_dir))
    results = topk_ids_scores(r, q, k=k)
    typer.echo(json.dumps({"query": q, "results": results}, ensure_ascii=False, indent=2))

def main():
    app()

if __name__ == "__main__":
    main()
