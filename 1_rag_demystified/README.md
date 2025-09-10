# RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters

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
    <a href="https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848"><img src="https://img.shields.io/badge/Medium-RAG%20Demystified-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>

</p>

This folder contains the code for [**RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters**](https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848) of the RAG Blog Series.  
It introduces the concept of Retrieval-Augmented Generation (RAG) and shows why retrieval matters by building a simple keyword-based Q&A system.

## What does `keyword_rag.py` do?
This script demonstrates a toy keyword-based Q&A system. It simulates a knowledge base, performs keyword search over documents, and generates answers using retrieved context. This helps illustrate why retrieval is important for grounding LLM responses.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Usage

To run the demo script:

```bash
python keyword_rag.py
```

Sample output:

```
Answering based on retrieved context: RAG stands for Retrieval-Augmented Generation. It helps LLMs fetch relevant context before answering. In enterprise settings, RAG reduces compliance risk by grounding answers in real company knowledge.
```

You can modify the question in the script to test other queries.
