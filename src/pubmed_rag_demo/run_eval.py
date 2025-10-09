from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
import typer
from .index import build_bm25_from_dir
from .eval import context_hit_rate, retrieval_precision_at_k, faithfulness_overlap
from .llm import LLM

app = typer.Typer(add_completion=False)
_WORD = re.compile(r"[a-z0-9]+")

def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))

def naive_candidates(qa_pairs, retriever, k_for_candidate: int = 1) -> list[str]:
    candidates: list[str] = []
    for q, _ in qa_pairs:
        results = retriever.query(q, k=k_for_candidate)
        if not results:
            candidates.append("")
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
        candidates.append(best_sent)
    return candidates

def llm_candidates(qa_pairs, retriever, model_name: str = "gpt-4o-mini") -> list[str]:
    llm = LLM(model=model_name)
    cands: list[str] = []
    for q, _ in qa_pairs:
        results = retriever.query(q, k=1)
        if not results:
            cands.append("")
            continue
        doc_id, _ = results[0]
        idx = retriever._doc_ids.index(doc_id)
        context = retriever._docs[idx]
        cands.append(llm.answer(q, context))
    return cands

@app.command()
def main(
    data_dir: str = typer.Argument(..., help="Folder with *.txt abstracts"),
    qa_path: str = typer.Argument(..., help="JSONL with fields: question, answer"),
    k: int = typer.Option(3, "--k", "-k", help="Top-k to retrieve"),
    out_dir: str = typer.Option("outputs", "--out-dir", help="Where to save metrics/report"),
    use_llm: bool = typer.Option(False, "--use-llm/--no-llm", help="Use LLM for candidates"),
    llm_model: str = typer.Option("gpt-4o-mini", "--llm-model", help="LLM model name"),
    dump_candidates: bool = typer.Option(False, "--dump-candidates/--no-dump-candidates", help="Write per-QA JSONL"),
):
    data_p = Path(data_dir)
    qa_p = Path(qa_path)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    typer.echo(f"DEBUG out_dir={out_p.resolve()} dump_candidates={dump_candidates}")

    # load QA
    qa_pairs = []
    with qa_p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qa_pairs.append((obj["question"], obj["answer"]))

    retriever = build_bm25_from_dir(data_p)

    # retrieval metrics
    hit = context_hit_rate(qa_pairs, retriever, k=k)
    prec = retrieval_precision_at_k(qa_pairs, retriever, k=k)

    # candidates
    if use_llm:
        cands = llm_candidates(qa_pairs, retriever, model_name=llm_model)
    else:
        cands = naive_candidates(qa_pairs, retriever, k_for_candidate=1)

    # faithfulness
    faith = faithfulness_overlap(qa_pairs, cands, retriever, k=k, threshold=0.6)

    # collate + save aggregate
    metrics = {"k": k, **hit, **prec, **faith, "used_llm": use_llm}
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
        f.write(f"- `used_llm`: **{use_llm}**\n")

    typer.echo(f"Saved metrics: {metrics_path}")
    typer.echo(f"Saved report:  {report_path}")
    typer.echo(f"dump_candidates flag = {dump_candidates}")

    # Per-QA dump
    if dump_candidates:
        dump_path = out_p / f"candidates_{ts}.jsonl"
        with dump_path.open("w", encoding="utf-8") as f:
            for (q, a_gt), cand in zip(qa_pairs, cands):
                results = retriever.query(q, k=1)
                doc_id, _ = results[0] if results else ("", 0.0)
                ctx = ""
                if results:
                    idx = retriever._doc_ids.index(doc_id)
                    ctx = retriever._docs[idx]
                f.write(json.dumps({
                    "question": q,
                    "answer_gt": a_gt,
                    "candidate": cand,
                    "top_doc_id": doc_id,
                    "top_doc_text": ctx[:1500]
                }, ensure_ascii=False) + "\n")
        typer.echo(f"Saved candidates: {dump_path}")

if __name__ == "__main__":
    app()
