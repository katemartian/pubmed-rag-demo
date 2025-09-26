from pubmed_rag_demo.corpus import load_txt_corpus

def test_load_txt_corpus_reads_nonempty_txt_files(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    empty = tmp_path / "empty.txt"
    a.write_text("Insulin lowers blood glucose.", encoding="utf-8")
    b.write_text("MRI can detect structural brain changes.", encoding="utf-8")
    empty.write_text("   ", encoding="utf-8")  # ignored

    docs, ids = load_txt_corpus(tmp_path)
    assert len(docs) == 2
    assert set(ids) == {"a", "b"}
    assert "Insulin" in docs[0] or "Insulin" in docs[1]
