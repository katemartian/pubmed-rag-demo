# Tiny RAG demo on PubMed abstracts with a simple evaluation harness.

## Corpus setup

Put your PubMed abstracts as plain text files into the `data/` folder.
- Each file = one abstract
- Filename (without `.txt`) becomes the document ID
- Example:
  - data/insulin.txt
  - data/mri.txt
- `data/` is in `.gitignore` so local corpora won’t be committed

### Try it out

```bash
python -m pubmed_rag_demo.cli ./data -q "insulin regulates glucose" -k 2
```
This will return top-k document IDs with BM25 scores in JSON.