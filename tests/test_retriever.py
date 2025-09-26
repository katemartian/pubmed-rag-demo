from pubmed_rag_demo.retriever import BM25Retriever

def test_bm25_retrieves_relevant_doc():
    docs = [
        "Insulin therapy helps regulate blood glucose in patients with diabetes.",
        "MRI imaging of the brain reveals structural changes in neurodegenerative disease.",
        "Antibiotic stewardship reduces resistance and improves clinical outcomes."
    ]
    r = BM25Retriever()
    r.add(docs, ids=["insulin", "mri", "antibiotic"])
    results = r.query("insulin regulates glucose levels", k=2)
    top_ids = [rid for rid, _ in results]
    assert "insulin" in top_ids
    # the insulin doc should rank above unrelated ones
    assert top_ids[0] == "insulin"
