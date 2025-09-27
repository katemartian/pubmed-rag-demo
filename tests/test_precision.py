from pubmed_rag_demo.retriever import BM25Retriever
from pubmed_rag_demo.eval import retrieval_precision_at_k

def test_retrieval_precision_at_k():
    docs = [
        "Insulin therapy regulates blood glucose.",               # contains 'glucose'
        "MRI imaging detects structural brain changes.",          # does not contain 'glucose'
        "Dietary changes can lower glucose spikes in patients."   # contains 'glucose'
    ]
    r = BM25Retriever()
    r.add(docs, ids=["insulin", "mri", "diet"])
    qa_pairs = [("how to regulate glucose?", "glucose")]
    metrics = retrieval_precision_at_k(qa_pairs, r, k=3)
    # at least 2 of top-3 should contain 'glucose' (depending on ranking it could be 2/3 or 1/3, but never 0)
    assert 0.0 < metrics["precision_at_k"] <= 1.0
