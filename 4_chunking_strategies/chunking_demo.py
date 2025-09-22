"""
Chunking Strategies Demo
Compare fixed, overlap, recursive, and semantic chunking on a sample PDF.
"""

import os
from typing import List
from PyPDF2 import PdfReader
import tiktoken
from sentence_transformers import SentenceTransformer, util


# --- Token counter: helps us measure chunk sizes properly ---
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a given text string.
    Falls back to cl100k_base if the specific model encoding is unavailable.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# --- Load PDF: extract raw text from the paper ---
def load_pdf(path: str) -> str:
    """
    Extract raw text from a PDF file.
    """
    reader = PdfReader(path)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())


# --- Fixed-size chunking: simple, but may cut through meaning ---
def chunk_fixed(text: str, chunk_size: int = 200, overlap: int = 0) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# --- Recursive chunking: preserves coherence by breaking down hierarchically ---
def chunk_recursive(text: str, max_tokens: int = 200) -> List[str]:
    """Break by paragraphs → sentences → tokens until under limit."""
    paragraphs = text.split("\n\n")
    result = []
    for para in paragraphs:
        if count_tokens(para) <= max_tokens:
            result.append(para)
        else:
            sentences = para.split(". ")
            buf = ""
            for sent in sentences:
                if count_tokens(buf + sent) > max_tokens:
                    result.append(buf.strip())  # close the buffer when too large
                    buf = sent
                else:
                    buf += " " + sent
            if buf:
                result.append(buf.strip())
    return result


# --- Semantic chunking: uses embeddings to split where topics naturally shift ---
def chunk_semantic(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.6
) -> List[str]:
    """Split text into semantically coherent chunks."""
    model = SentenceTransformer(model_name)
    sentences = text.split(". ")  # Split text into sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)  # Get embeddings for each sentence

    chunks, buf = [], [sentences[0]]  # Initialize chunks and buffer with first sentence
    for i in range(1, len(sentences)):
        # Compute cosine similarity between the current and previous sentence embeddings
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        if sim < threshold:  # If similarity drops below threshold, start a new chunk
            chunks.append(". ".join(buf))  # Add current buffer as a chunk
            buf = [sentences[i]]  # Start new buffer with current sentence
        else:
            buf.append(sentences[i])  # Continue adding to current buffer
    if buf:
        chunks.append(". ".join(buf))  # Add any remaining sentences as the last chunk
    return chunks


# --- Demo run: compare chunking strategies ---
if __name__ == "__main__":
    pdf_path = os.path.join("data", "sample_document.pdf")
    pdf_text = load_pdf(pdf_path)

    fixed_chunks = chunk_fixed(pdf_text, 200)
    overlap_chunks = chunk_fixed(pdf_text, 200, overlap=50)
    recursive_chunks = chunk_recursive(pdf_text, 200)
    semantic_chunks = chunk_semantic(pdf_text, threshold=0.65)

    print("Fixed:", len(fixed_chunks))
    print("Overlap:", len(overlap_chunks))
    print("Recursive:", len(recursive_chunks))
    print("Semantic:", len(semantic_chunks))
