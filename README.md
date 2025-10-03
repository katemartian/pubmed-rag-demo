# Tiny RAG demo on PubMed abstracts with a simple evaluation harness.

[![CI](https://github.com/katemartian/pubmed-rag-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/katemartian/pubmed-rag-demo/actions/workflows/ci.yml)

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

## Leaderboard (auto-generated)

<!-- LB-START -->

| k | hit_rate | precision@k | faithfulness |
|---|----------|-------------|--------------|
| 1 | 0.500 | 0.500 | 1.000 |
| 2 | 1.000 | 0.500 | 1.000 |
| 3 | 1.000 | 0.333 | 1.000 |

<!-- LB-END -->
