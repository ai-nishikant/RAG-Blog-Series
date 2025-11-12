# 📘 Building Your First RAG Pipeline — Code Companion  


<!-- Center-aligned professional badges -->
<p align="center">

  <!-- First Line: Repo Metadata + GitHub Engagement (All Blue) -->
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/repo-size/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Repo%20Size&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/languages/top/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=python&label=Python&color=blue"></a>
  
<!-- Second Line: Blog & Social (Brand Colors) -->
  <a href="https://medium.com/@ai.nishikant"><img src="https://img.shields.io/badge/Medium-Read%20My%20Blog-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
  <a href="https://medium.com/@ai.nishikant/building-your-first-rag-pipeline-5de171f4cf8f"><img src="https://img.shields.io/badge/Medium-Building%20Your%20First%20RAG%20Pipeline-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>  
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>

</p>

This repository contains the code companion for the blog:  
**“Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers”**  

It provides a simple end-to-end example of **Retrieval-Augmented Generation (RAG)**:  
- Extract text from a PDF  
- Chunk text into smaller pieces  
- Build a FAISS index of embeddings  
- Retrieve the most relevant chunks  
- Query an LLM (OpenAI or Groq) with context to generate reliable answers  

---

## 📂 Project Structure  

```
3_first_rag_pipeline/
├── main_openai.py      # RAG pipeline using OpenAI API
├── main_groq.py        # RAG pipeline using Groq API
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
├── sample.pdf          # Example input file (replace with your own)
└── README.md           # This file
```

---

## ⚙️ Setup  

### 1. Clone the repo and enter the folder  
```bash
git clone https://github.com/ai-nishikant/RAG-Blog-Series.git
cd RAG-Blog-Series/3_first_rag_pipeline
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Configure API keys  
Copy `.env.example` to `.env` and update with your API key(s):  

```bash
cp .env.example .env
```

Fill in either:  

- `OPENAI_API_KEY` for **OpenAI**  
- `GROQ_API_KEY` for **Groq**  

---

## ▶️ Run the pipeline  

### OpenAI version  
```bash
python main_openai.py
```

### Groq version  
```bash
python main_groq.py
```

Both scripts expect a file named `sample.pdf` in the folder. You can replace it with any other PDF.  

---

## 📝 Notes  

- The pipeline defaults to **500-word chunks** and retrieves the **top-3 chunks** for each query.  
- Default models:  
  - OpenAI → `gpt-4o-mini`  
  - Groq → `llama-3.1-8b-instant`  
- Change these defaults by editing `.env`.  
- If you’re on Windows and `faiss-cpu` fails to install, try using WSL or Conda for compatibility.  

> **If you’d like to try this pipeline with free Groq API keys, see the GitHub repo for an alternative implementation.**  

---

## 🔧 Troubleshooting  

Here are some common issues and how to fix them:  

1. **`ModuleNotFoundError: No module named 'faiss'`**  
   - Ensure you installed `faiss-cpu` (not `faiss`).  
   - Run:  
     ```bash
     pip install faiss-cpu
     ```  

2. **FAISS install fails on Windows**  
   - Try using **Python 3.10 or 3.11** (some wheels don’t support latest versions).  
   - Alternatively, use **Conda**:  
     ```bash
     conda install -c conda-forge faiss-cpu
     ```  

3. **`AttributeError: 'ChatCompletion' object has no attribute 'message'`**  
   - You may be using an outdated SDK. Update:  
     ```bash
     pip install --upgrade openai groq
     ```  

4. **`RuntimeError: LLM query failed`**  
   - Check your `.env` file is set correctly with `OPENAI_API_KEY` or `GROQ_API_KEY`.  
   - Ensure you copied `.env.example` → `.env` and reloaded your shell.  

5. **API quota or authentication errors**  
   - OpenAI free tier may expire quickly.  
   - Groq offers free API keys — try switching to Groq if OpenAI credits are exhausted.  

6. **Sentence-Transformers download too slow / stuck**  
   - Models are fetched the first time from HuggingFace. If blocked, use a VPN or pre-download.  
   - Try:  
     ```bash
     from sentence_transformers import SentenceTransformer
     SentenceTransformer("all-MiniLM-L6-v2")
     ```  
     to force a local download.  

7. **Text extraction from PDF is empty**  
   - Some PDFs are scanned images without embedded text.  
   - Try using OCR (e.g., `pytesseract`) before running this pipeline.  

---

## 🔗 Blog Series

This companion belongs to the **RAG Blog Series**:  
1. [RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters](https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848)  
2. [Speaking in Vectors: A Beginner’s Guide to Embeddings and Semantic Search](https://medium.com/@ai.nishikant/speaking-in-vectors-1b8142f9ec87)  
3. [Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers](https://medium.com/@ai.nishikant/building-your-first-rag-pipeline-5de171f4cf8f)  
4. [Better Chunks, Better Answers: Chunking Strategies for Smarter RAG](https://medium.com/@ai.nishikant/rag-chunking-strategies-ba414704c33e)  
5. [How to Optimize RAG Context Windows for Smarter Retrieval](https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-b26859f03b2d)  
6. [Smarter Prompts: Engineering Better Instructions in RAG](https://medium.com/@ai.nishikant/smarter-prompts-engineering-better-instructions-in-rag-58e87ad8077f)
