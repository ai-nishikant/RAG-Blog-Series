import os
import time
from typing import List

def read_texts_from_dir(path: str, limit: int | None = None) -> List[str]:
    files = sorted([f for f in os.listdir(path) if f.endswith(".txt")])
    texts = []
    for fname in files[: limit or len(files)]:
        with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
        self.elapsed = self.end - self.start

def sort_by_recency_filenames(path: str, limit: int = 3) -> List[str]:
    """
    Expects filenames like YYYY-MM-DD_title.txt and sorts descending by date.
    Returns file contents in recency order.
    """
    files = [f for f in os.listdir(path) if f.endswith(".txt")]
    files.sort(reverse=True)  # lexical works for YYYY-MM-DD prefix
    texts = []
    for fname in files[:limit]:
        with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    return texts
