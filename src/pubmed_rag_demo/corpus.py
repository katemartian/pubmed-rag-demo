from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

def load_txt_corpus(dir_path: str | Path) -> Tuple[List[str], List[str]]:
    """
    Load all *.txt files under dir_path (non-recursive).
    Returns (docs, ids) where ids are filenames without extension.
    Ignores empty files.
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    docs: List[str] = []
    ids: List[str] = []
    for fp in sorted(p.glob("*.txt")):
        text = fp.read_text(encoding="utf-8").strip()
        if text:
            docs.append(text)
            ids.append(fp.stem)
    return docs, ids
