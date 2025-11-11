# 📘 Smarter Prompts in RAG — Code Companion

<p align="center">

  <!-- Repo Metadata -->
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/repo-size/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Repo%20Size&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/languages/top/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=python&label=Python&color=blue"></a>
  <!-- Blog & Social -->
  <a href="https://medium.com/@ai.nishikant"><img src="https://img.shields.io/badge/Medium-Read%20My%20Blog-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
  <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>
</p>

This repository contains the code companion for the blog:  
**“Smarter Prompts: Engineering Better Instructions in RAG.”**

It demonstrates compact, runnable examples of prompt strategies discussed in the post. Each script is intentionally small and self-contained—designed for clarity over production complexity.

- Zero-shot vs Few-shot prompting  
- Chain-of-Thought prompting  
- Self-Consistency prompting  
- Instruction Layering (global + task directives)  
- Retrieval-Aware prompting (recency-weighted toy example)  
- Constraint-based prompting (JSON schema)

---

## 📂 Project Structure

```
6_smarter_prompts/
├── scripts/
│   ├── zero_vs_fewshot.py
│   ├── chain_of_thought.py
│   ├── self_consistency.py
│   ├── instruction_layering.py
│   ├── retrieval_aware.py
│   └── constraint_json.py
├── data/
│   ├── support_tickets/
│   ├── compliance_policies/
│   ├── vendor_risk/
│   ├── news/
│   └── contracts/
├── prompts/
│   ├── templates.yaml
│   └── few_shot_examples.yaml
├── src/
│   ├── llm_client.py
│   ├── utils.py
│   └── eval_metrics.py
├── outputs/
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1) Clone and enter
```bash
git clone https://github.com/ai-nishikant/RAG-Blog-Series.git
cd RAG-Blog-Series/6_prompts_companion
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) API key
Copy and edit the environment file:
```bash
cp .env.example .env
```
Add your key, for example:
- `OPENAI_API_KEY=...`

(Optional) If your SDK returns token usage, scripts will log it; otherwise print “NA”.

---

## ▶️ Quickstart (Run Any Demo)

Each script loads a tiny dataset, renders a prompt template, calls the model once (or a few times), prints the answer, and saves JSONL output.

### A) Zero-shot vs Few-shot
```bash
python -m scripts.zero_vs_fewshot
```

### B) Chain-of-Thought
```bash
python -m scripts.chain_of_thought
```

### C) Self-Consistency
```bash
python -m scripts.self_consistency
```

### D) Instruction Layering
```bash
python -m scripts.instruction_layering
```

### E) Retrieval-Aware
```bash
python -m scripts.retrieval_aware
```

### F) Constraint-based (JSON)
```bash
python -m scripts.constraint_json
```

---

## 🧪 Metrics and Outputs

- **ROUGE-1 / ROUGE-L** (optional)
- **Latency** and **token usage**


---

## 📝 Notes

- These demos favor clarity, not production readiness.  
- Datasets are toy-sized but realistic.  
- Adjust `src/llm_client.py` to use your chosen provider.

---

## 🔧 Troubleshooting

1) Run from inside `6_smarter_prompts/`.  
2) Add your API key in `.env`.  
3) Token usage may show NA if SDK doesn’t expose it.  
4) Constraint validation errors print inline.

---

## 🔗 Blog Series

This companion belongs to the **RAG Blog Series**:  
1. [RAG Demystified — A Beginner’s Guide to What It Is and Why It Matters](https://medium.com/@ai.nishikant/rag-demystified-a-beginners-guide-to-what-it-is-and-why-it-matters-8df6a7388848)  
2. [Speaking in Vectors: A Beginner’s Guide to Embeddings and Semantic Search](https://medium.com/@ai.nishikant/speaking-in-vectors-1b8142f9ec87)  
3. [Building Your First RAG Pipeline: How Retrieval + LLMs Deliver Reliable Answers](https://medium.com/@ai.nishikant/building-your-first-rag-pipeline-5de171f4cf8f)  
4. [Better Chunks, Better Answers: Chunking Strategies for Smarter RAG](https://medium.com/@ai.nishikant/rag-chunking-strategies-ba414704c33e)  
5. [How to Optimize RAG Context Windows for Smarter Retrieval](https://medium.com/@ai.nishikant/how-to-optimize-rag-context-windows-for-smarter-retrieval)  
6. [Smarter Prompts: Engineering Better Instructions in RAG](https://medium.com/@ai.nishikant/smarter-prompts-engineering-better-instructions-in-rag-58e87ad8077f)
