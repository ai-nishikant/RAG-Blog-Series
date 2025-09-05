# RAG Blog Series
Code companion for the RAG Blog Series. Each folder corresponds to a blog post with its own README, requirements, and code, guiding you from keyword search to scalable, production-ready RAG systems.

# RAG Blog Series — Code Companion

Code companion for the **RAG Blog Series**.  
Each folder corresponds to a blog post with its own README, requirements, and runnable code.  
The series walks you through the journey of **Retrieval-Augmented Generation (RAG)** — from simple keyword search all the way to scalable, production-ready RAG systems.

---

## 📚 Blog Series Index

- [Blog 1 – RAG Demystified: A Beginner’s Guide to What It Is and Why It Matters](./1_rag_demystified/)
  
  Learn why LLMs hallucinate, why retrieval matters, and build a toy keyword-based Q&A system.  

- [Blog 2 – Speaking in Vectors: A Beginner's Guide to Embeddings and Semantic Search](./2_speaking_in_vectors/)  
  LUnderstanding embeddings, cosine similarity, and FAISS - the building blocks of smarter RAG systems.  

(More coming soon: evaluation, monitoring, observability, deployment…)

---

## How to Use This Repo

Each blog folder is independent.  
To run code for a specific blog:

```bash
cd 1_rag_demystified    
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python toy_rag.py       # or the relevant script for that blog

