# RAG Blog Series

<!-- Center-aligned professional badges -->
<p align="center">

  <!-- First Line: Repo Metadata + GitHub Engagement (All Blue) -->
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/repo-size/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Repo%20Size&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/languages/top/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=python&label=Python&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series/commits/main"><img src="https://img.shields.io/github/last-commit/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Last%20Commit&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series/stargazers"><img src="https://img.shields.io/github/stars/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Stars&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series/network/members"><img src="https://img.shields.io/github/forks/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Forks&color=blue"></a>
<!-- Second Line: Blog & Social (Brand Colors) -->
  <a href="https://medium.com/@ai.nishikant"><img src="https://img.shields.io/badge/Medium-Read%20My%20Blog-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>

</p>

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
  Understanding embeddings, cosine similarity, and FAISS - the building blocks of smarter RAG systems.  

- [Blog 3 – Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers](./3_first_rag_pipeline/)  
  Build your first RAG pipeline with retrieval + LLMs and turn documents into reliable, grounded AI answer

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

