# Speaking in Vectors — Companion Code

This repository provides the companion code for the blog post:

 [**Speaking in Vectors: A Beginner’s Guide to Embeddings and Semantic Search**](https://medium.com/@ai.nishikant/speaking-in-vectors-1b8142f9ec87) 

It includes a simple yet powerful implementation of semantic search using:

- **Sentence-Transformers** for embedding text  
- **FAISS** for efficient vector similarity search

---

## Quickstart

Follow these steps to run the code:

    git clone https://github.com/your-username/rag-embeddings-companion.git
    cd rag-embeddings-companion
    pip install -r requirements.txt
    python semantic_search.py

---

## What’s Inside

- **`semantic_search.py`**  
  Contains:
  - Model initialization (`all-MiniLM-L6-v2`)
  - A sample list of documents  
  - FAISS index creation using `IndexFlatL2`  
  - `semantic_search()` function for retrieving top-k similar documents  
  - Executable main block that runs an example search

- **`requirements.txt`**  
  Lists the dependencies:

        sentence-transformers
        faiss-cpu
        numpy

---

## How It Works

1. **Load documents** – Sample sentences about different meanings of *jaguar*.  
2. **Generate embeddings** – Documents are converted into vector representations.  
3. **Index embeddings** – FAISS (`IndexFlatL2`) stores and enables fast nearest-neighbor search.  
4. **Search queries** – Embed a text query and retrieve the most semantically similar documents.

This script covers embedding, indexing, and semantic search—all in just a few lines of readable code.

---

## Customization Ideas

Feel free to experiment by:

- Replacing `documents` with your own content (like PDFs, wiki articles)  
- Adjusting `top_k` in `semantic_search()` to retrieve more results  
- Adding chunking to handle longer documents  
- Swapping in a more scalable FAISS index (like IVF or HNSW) as your data grows

---

## Usage Example

Run the script:

    python semantic_search.py

You may see output like:

    Search results:
    - The jaguar is a large cat native to the Americas. (distance=0.1234)
    - Tigers are also big cats found in Asia. (distance=0.2345)

Try modifying the `query` or the `documents` to see how semantic relevance shifts.

---

## Related Resources

- [Sentence-Transformers Documentation](https://www.sbert.net/)  
- [FAISS Guide: Choosing an Index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)  
- Look forward to upcoming blog posts that will expand this into a full RAG pipeline with LLM integration.

---

