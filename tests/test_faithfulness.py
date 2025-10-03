from pubmed_rag_demo.retriever import BM25Retriever
from pubmed_rag_demo.eval import faithfulness_overlap

def test_faithfulness_overlap_simple():
    docs = [
        "Insulin therapy helps regulate blood glucose in diabetes.",
        "MRI imaging detects structural brain changes."
    ]
    r = BM25Retriever()
    r.add(docs, ids=["insulin", "mri"])
    qa_pairs = [("what regulates glucose?", "insulin")]
    candidates = ["insulin therapy regulates blood glucose"]  # supported by doc 0
    metrics = faithfulness_overlap(qa_pairs, candidates, r, k=2, threshold=0.5)
    assert 0.0 < metrics["faithfulness"] <= 1.0

def test_faithfulness_overlap_flags_hallucination():
    docs = ["MRI imaging detects brain changes."]
    r = BM25Retriever()
    r.add(docs, ids=["mri"])
    qa_pairs = [("what detects brain changes?", "mri")]
    candidates = ["banana therapy cures glucose"]  # nonsense not in evidence
    metrics = faithfulness_overlap(qa_pairs, candidates, r, k=1, threshold=0.5)
    assert metrics["faithfulness"] == 0.0
