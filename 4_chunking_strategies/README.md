# 📘 Chunking Strategies for Smarter RAG — Code Companion  

<!-- Center-aligned professional badges -->
<p align="center">

  <!-- First Line: Repo Metadata + GitHub Engagement (All Blue) -->
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/repo-size/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Repo%20Size&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/languages/top/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=python&label=Python&color=blue"></a>
  <!-- Second Line: Blog & Social (Brand Colors) -->
  <a href="https://medium.com/@ai.nishikant"><img src="https://img.shields.io/badge/Medium-Read%20My%20Blog-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
  <a href="https://medium.com/@ai.nishikant/better-chunks-better-answers-chunking-strategies-for-smarter-rag"><img src="https://img.shields.io/badge/Medium-Chunking%20Strategies%20for%20Smarter%20RAG-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>  
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>

</p>

This repository contains the code companion for the blog:  
**“Better Chunks, Better Answers: Chunking Strategies for Smarter RAG”**  

It demonstrates how different chunking strategies affect retrieval quality in **Retrieval-Augmented Generation (RAG)**:  
- Fixed-size splitting  
- Overlap windows  
- Recursive splitting  
- Semantic splitting (embedding-based)  

---

## 📂 Project Structure  

```
4_chunking_strategies/
├── chunking_demo.py        # Compare fixed, overlap, recursive, and semantic chunking
├── requirements.txt        # Python dependencies
├── data/
│   ├── sample_document.pdf # Default test file (ships with repo)
└── README.md               # This file
```

---

## ⚙️ Setup  

### 1. Clone the repo and enter the folder  
```bash
git clone https://github.com/ai-nishikant/RAG-Blog-Series.git
cd RAG-Blog-Series/4_chunking_strategies
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

> **Note:** No API keys required. All code runs locally.  
> Sentence-Transformers will automatically download the `all-MiniLM-L6-v2` model on first use.  

---

## ▶️ Run the demo  

Run the default chunking demo on the included **sample document**:  
```bash
python chunking_demo.py
```

Expected output (chunk counts):  
```
Fixed: XX
Overlap: XX
Recursive: XX
Semantic: XX
```

You can also replace `sample_document.pdf` with your own PDF by placing it in `data/` and updating the path in `chunking_demo.py`.  

---

## 📝 Notes  

- **Default file:** `sample_document.pdf` is a short synthetic paper designed to show clear section boundaries, varying paragraph lengths, and topic transitions.  

- **Strategies:**  
  - Fixed-size → simple but may cut sentences.  
  - Overlap → preserves context across boundaries at higher cost.  
  - Recursive → ensures no chunk exceeds token limit.  
  - Semantic → groups related sentences by meaning (extra compute).  

---

## 🔧 Troubleshooting  

1. **`ModuleNotFoundError: No module named 'tiktoken'`**  
   - Ensure you ran `pip install -r requirements.txt`.  

2. **Sentence-Transformers model download too slow**  
   - HuggingFace models are downloaded the first time.  
   - If blocked, use a VPN or pre-download with:  
     ```python
     from sentence_transformers import SentenceTransformer
     SentenceTransformer("all-MiniLM-L6-v2")
     ```  

3. **PDF extraction returns empty text**  
   - Some PDFs are scanned images, not digital text.  
   - Use OCR first (e.g., `pytesseract`) before running chunking.  

---

## 🔗 Blog Series

This companion belongs to the **RAG Blog Series**:  
1. [RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters](https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848)  
2. [Speaking in Vectors: A Beginner’s Guide to Embeddings and Semantic Search](https://medium.com/@ai.nishikant/speaking-in-vectors-1b8142f9ec87)  
3. [Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers](https://medium.com/@ai.nishikant/building-your-first-rag-pipeline-5de171f4cf8f)  
4. [Better Chunks, Better Answers: Chunking Strategies for Smarter RAG](https://medium.com/@ai.nishikant/rag-chunking-strategies-ba414704c33e)  
5. [How to Optimize RAG Context Windows for Smarter Retrieval](https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-b26859f03b2d)  
6. [Smarter Prompts: Engineering Better Instructions in RAG](https://medium.com/@ai.nishikant/smarter-prompts-engineering-better-instructions-in-rag-58e87ad8077f)
