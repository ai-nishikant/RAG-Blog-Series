# 📘 How to Optimize RAG Context Windows for Smarter Retrieval — Code Companion  

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
  
  <a href="https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-for-smarter-retrieval"><img src="https://img.shields.io/badge/Medium-How%20to%20Optimize%20RAG%20Context%20Windows%20for%20Smarter%20Retrieval-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>  
  
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>

</p>

This repository contains the code companion for the blog:  
**“How to Optimize RAG Context Windows for Smarter Retrieval.”**  

It demonstrates practical techniques for optimizing retrieval and context management in **Retrieval-Augmented Generation (RAG)** systems:  
- Multi-stage retrieval  
- Multi-query rewriting  
- Reranking (Cohere or lexical fallback)  
- Dynamic metadata filtering  
- Summarization under tight context (flagship demo)  
- Hybrid pipelines combining multiple techniques  

---

## 📂 Project Structure  

```
5_context_windows/
├── scripts/
│   ├── 01_multistage.py
│   ├── 02_multiquery.py
│   ├── 03_rerank.py
│   ├── 04_filtering.py
│   ├── 05_summarize.py        # Flagship demo script
│   └── 06_hybrid.py
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
Copy the example environment file and edit it with your API keys:  
```bash
cp .env.example .env
```

Fill in as needed:
- `GROQ_API_KEY` for summarization and query rewriting  
- `COHERE_API_KEY` for reranking (optional)  
- `OPENAI_API_KEY` if you use OpenAI for comparison  

---

## ▶️ Run the demos  

### Flagship: Summarization under Tight Context  
```bash
python scripts/05_summarize.py --data_dir data/sample_docs --k 5 --use_llm
```

### Other techniques  
```bash
python scripts/01_multistage.py
python scripts/02_multiquery.py
python scripts/03_rerank.py
python scripts/04_filtering.py
python scripts/06_hybrid.py
```

Each script prints:
- Retrieved document IDs  
- Token counts before and after optimization  
- Time taken per step  

---

## 📝 Notes  

- All demos run locally and finish within seconds.  
- Groq API is used for summarization and rewriting (optional).  
- Cohere API can be enabled for reranking with `--use_cohere`.  
- Each script mirrors a section of the blog and is designed for clarity, not production use.  

---

## 🔧 Troubleshooting  

1. **`ModuleNotFoundError`** → Ensure you installed all dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Groq or Cohere API key missing** → Add to `.env` and reload your shell.

3. **Slow or stuck model downloads** → First-time download for `sentence-transformers`; retry after network stabilization.

4. **Empty retrieval results** → Try adjusting `--k` or modifying the sample corpus under `data/sample_docs/`.

---

## 🔗 Blog Series  

This is part of the **RAG Blog Series**:  
1. [RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters](https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848)  
2. [Speaking in Vectors: A Beginner’s Guide to Embeddings and Semantic Search](https://medium.com/@ai.nishikant/speaking-in-vectors-1b8142f9ec87)  
3. [Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers](https://medium.com/@ai.nishikant/building-your-first-rag-pipeline-5de171f4cf8f)  
4. [Better Chunks, Better Answers: Chunking Strategies for Smarter RAG](https://medium.com/@ai.nishikant/better-chunks-better-answers-chunking-strategies-for-smarter-rag)  
5. [How to Optimize RAG Context Windows for Smarter Retrieval](https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-for-smarter-retrieval)
