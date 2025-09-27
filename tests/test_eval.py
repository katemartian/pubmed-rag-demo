from pubmed_rag_demo.retriever import BM25Retriever
from pubmed_rag_demo.eval import context_hit_rate

def test_context_hit_rate_simple():
    docs = [
        "Insulin therapy regulates blood glucose.",
        "MRI imaging detects brain changes."
    ]
    r = BM25Retriever()
    r.add(docs, ids=["insulin", "mri"])
    qa_pairs = [("what regulates glucose?", "glucose")]
    metrics = context_hit_rate(qa_pairs, r, k=2)
    assert metrics["hit_rate"] == 1.0
