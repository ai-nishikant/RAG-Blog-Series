# RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters

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
