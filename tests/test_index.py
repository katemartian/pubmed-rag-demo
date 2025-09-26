from pubmed_rag_demo.index import build_bm25_from_dir, topk_ids_scores

def test_build_and_query(tmp_path):
    (tmp_path / "insulin.txt").write_text(
        "Insulin therapy helps regulate blood glucose.", encoding="utf-8"
    )
    (tmp_path / "mri.txt").write_text(
        "MRI imaging reveals structural brain changes.", encoding="utf-8"
    )

    r = build_bm25_from_dir(tmp_path)
    results = topk_ids_scores(r, "insulin regulates glucose", k=2)
    assert results, "should return at least one match"
    top_ids = [rid for rid, _ in results]
    assert top_ids[0] == "insulin"
