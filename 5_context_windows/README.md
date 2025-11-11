# 📘 How to Optimize RAG Context Windows for Smarter Retrieval — Code Companion  

<!-- Center-aligned professional badges -->
<p align="center">

  <!-- First Line: Repo Metadata + GitHub Engagement (All Blue) -->
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/repo-size/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Repo%20Size&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/languages/top/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=python&label=Python&color=blue"></a>

  <!-- Second Line: Blog & Social (Brand Colors) -->
  <a href="https://medium.com/@ai.nishikant"><img src="https://img.shields.io/badge/Medium-Read%20My%20Blog-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
  <a href="https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-for-smarter-retrieval"><img src="https://img.shields.io/badge/Medium-How%20to%20Optimize%20RAG%20Context%20Windows%20for%20Smarter%20Retrieval-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>  
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>

</p>

This repository contains the code companion for the blog:  
**“How to Optimize RAG Context Windows for Smarter Retrieval.”**  

It demonstrates multiple strategies for optimizing retrieval and managing context in **Retrieval-Augmented Generation (RAG)** systems — from candidate selection to compression:  
- Multi-stage retrieval  
- Multi-query rewriting  
- Reranking (Cohere or lexical fallback)  
- Dynamic metadata filtering  
- Summarization under tight context (flagship demo)  
- Hybrid orchestration combining all techniques  

---

## 📂 Project Structure  

```
5_context_windows/
├── scripts/
│   ├── 01_multistage.py        # Broad → refined retrieval
│   ├── 02_multiquery.py        # Multi-query rewriting (Groq or static)
│   ├── 03_rerank.py            # Cohere v3 or lexical fallback reranking
│   ├── 04_filtering.py         # Dynamic metadata filtering
│   ├── 05_summarize.py         # Flagship demo: summarization under tight context
│   └── 06_hybrid.py            # End-to-end pipeline combining all techniques
├── data/
│   ├── sample_docs/            # Tiny synthetic corpus for testing
│   └── queries.jsonl           # Example questions with relevance info
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup  

### 1. Clone the repo and enter the folder  
```bash
git clone https://github.com/ai-nishikant/RAG-Blog-Series.git
cd RAG-Blog-Series/5_context_windows
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Configure API keys  
Copy the example environment file and edit it with your keys:  
```bash
cp .env.example .env
```

Fill in as needed:  
- `GROQ_API_KEY` → summarization and query rewriting  
- `COHERE_API_KEY` → reranking (optional)  
- `OPENAI_API_KEY` → comparison or testing  

Use the current supported model:  
```bash
GROQ_MODEL=llama-3.1-8b-instant
```

---

## ▶️ Hands-On Demos  

### Flagship: Summarization under Tight Context (Groq)
Compare “small” vs “large” context plans:

```bash
# Small window (forces summarization)
python scripts/05_summarize.py   --question "Summarize the evaluation approach and key results."   --data_dir data/sample_docs   --k 5   --plan small   --use_llm

# Large window (uses full context)
python scripts/05_summarize.py   --question "Summarize the evaluation approach and key results."   --data_dir data/sample_docs   --k 8   --plan large
```

### Explore Other Techniques  
You can also run each supporting script individually to explore specific retrieval optimizations:

```bash
python scripts/01_multistage.py        # Multi-stage retrieval
python scripts/02_multiquery.py        # Multi-query rewriting
python scripts/03_rerank.py            # Reranking (Cohere/lexical)
python scripts/04_filtering.py         # Metadata filtering
python scripts/06_hybrid.py            # End-to-end hybrid pipeline
```

Each prints:
- Retrieved document IDs  
- Token counts before and after optimization  
- Context composition summaries  

---

## 📝 Notes  

- All demos run locally and complete within seconds.  
- LLM calls (Groq, Cohere) are **optional** and safely degrade to local fallbacks.  
- Each script corresponds to a core concept in the blog, designed for **clarity, not production**.  
- Token counts use `tiktoken` for approximate context measurement.  

---

## 🔧 Troubleshooting  

1. **`data_dir not found`** → Ensure you run scripts from inside `5_context_windows/`.  
2. **Groq model error (`model_decommissioned`)** → Update to `llama-3.1-8b-instant`.  
3. **Cohere key missing** → Add `COHERE_API_KEY` in `.env`.  
4. **Slow model downloads** → First-time setup for `sentence-transformers`.  

---

## 🔗 Blog Series  

This is part of the **RAG Blog Series**:  
1. [RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters](https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848)  
2. [Speaking in Vectors: A Beginner’s Guide to Embeddings and Semantic Search](https://medium.com/@ai.nishikant/speaking-in-vectors-1b8142f9ec87)  
3. [Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers](https://medium.com/@ai.nishikant/building-your-first-rag-pipeline-5de171f4cf8f)  
4. [Better Chunks, Better Answers: Chunking Strategies for Smarter RAG](https://medium.com/@ai.nishikant/rag-chunking-strategies-ba414704c33e)  
5. [How to Optimize RAG Context Windows for Smarter Retrieval](https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-for-smarter-retrieval)  
