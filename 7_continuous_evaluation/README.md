# Continuous Evaluation for RAG Systems  

<div align="center">

  <!-- Repo Metadata -->
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/repo-size/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=github&label=Repo%20Size&color=blue"></a>
  <a href="https://github.com/ai-nishikant/RAG-Blog-Series"><img src="https://img.shields.io/github/languages/top/ai-nishikant/RAG-Blog-Series?style=flat-square&logo=python&label=Python&color=blue"></a>
  <!-- Blog & Social -->
  <a href="https://medium.com/@ai.nishikant"><img src="https://img.shields.io/badge/Medium-Read%20My%20Blog-00ab6c?style=flat-square&logo=medium&logoColor=white"></a>
   <a href="https://linkedin.com/in/nishikant-surwade"><img src="https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-0077B5?style=flat-square&logo=linkedin&logoColor=white"></a>
</div>


Companion Code for the Blog:  
**“Why RAG Systems Drift — And How Continuous Evaluation Keeps Them Stable”**

This repository contains the companion code for the seventh article in the RAG Blog Series.  
It implements the foundations of a **continuous evaluation pipeline** for RAG systems, focusing on:

- capturing instrumentation signals
- measuring intrinsic / extrinsic / behavioral metrics
- preparing these signals for multi-stage evaluation
- building toward controllers and feedback loops discussed in the blog

This repo is structured so readers can run, extend, and adapt the ideas from the article in a real project environment.

---

## About This Repository

The blog presents simplified code snippets for clarity.  
This repository contains **fully runnable, production-friendly versions** of those snippets:

- modular functions  
- type hints and docstrings  
- input validation  
- optional configuration-driven behavior  
- notebooks demonstrating real usage  

The goal is to bridge the gap between **blog explanation** and **engineering implementation**.

This code is step-aligned with the blog’s sections:

| Blog Step | Purpose | Implemented In |
|----------|----------|----------------|
| Step 1 — Instrumentation | Capture evaluation signals | `src/rag_eval/metrics.py` |
| Step 2 — Multi-Stage Evaluation Loop | Localize failures (A/B/C) | `stages.py` |
| Step 3 — Evaluation Controller | Real-time corrective actions | `controller.py` |
| Step 4 — Feedback Loops | Offline/online continuous improvement | `feedback_loops.py` |
| Step 5 — Engineering Patterns | Canary prompts, shadow eval, audits | `patterns.py` |
| Hands-On |Minimal runnable demo|  Notebooks 01–04 |
| Demo | Demo | `run_demo.py` |
| Dashboard | Dashboard| `evaluation_dashboard_mock.json` |
| Synthetic Drift | Data | `scenarios.py`, `demo_scenarios.yaml` |
| Hands-On |Minimal runnable demo|  Notebooks 01–04 |
| Demo |Demo | `run_demo.py` |
| Dashboard | Dashboard| `evaluation_dashboard_mock.json` |


As future articles in the series continue building the system, this folder will grow to include:

- stage evaluation logic  
- controllers  
- feedback mechanisms  
- drift simulation tools  
- dashboard examples  

---

## Repository Structure

```
7_continuous_evaluation/
├── README.md
├── requirements.txt
├── .env.example
│
├── src/
│   └── rag_eval/
│       ├── __init__.py
│       ├── metrics.py
│       ├── stages.py
│       ├── controller.py
│       ├── feedback_loops.py
│       ├── patterns.py
│       └── scenarios.py
│
├── notebooks/
│   ├── 01_metrics_and_signals.ipynb
│   ├── 02_multi_stage_evaluation_loop.ipynb
│   ├── 03_controller_decision_logic.ipynb
│   └── 04_end_to_end_simulation.ipynb
│
├── config/
│   ├── evaluation_config.yaml
│   └── demo_scenarios.yaml
│
├── data/
│   ├── references/sample_reference_answers.json
│   ├── logs/sample_requests.jsonl
│   ├── logs/sample_evaluation_runs.jsonl
│   └── prompts/
│       ├── canary_prompts.yaml
│       └── diagnostic_queries.yaml
│
├── dashboards/
│   └── examples/
│       ├── evaluation_dashboard_mock.json
│       └── README.md
│
├── scripts/
│   └── run_demo.py
│
└── tests/
    ├── test_metrics.py
    ├── test_stages.py
    ├── test_controller.py
    ├── test_feedback_loops.py
    └── test_patterns.py
```

The major modules in repo (not required to run Step 1):

```
src/rag_eval/stages.py           # Stage A/B/C evaluation logic
src/rag_eval/controller.py       # Automated evaluation controller
src/rag_eval/feedback_loops.py   # Offline/online feedback signals
src/rag_eval/patterns.py         # Canary prompts, shadow eval, reasoning audits
config/evaluation_config.yaml    # Thresholds, metric weights, feature flags
data/prompts/                    # Canary + diagnostic prompts
dashboards/examples/             # Example dashboard schemas/payloads
tests/                            # Complete test suite
```

---

## Installation

From the root of the RAG Blog Series repository:

```bash
cd 7_continuous_evaluation
```

### (Optional) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```


# Getting Started

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Set Python path
```
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```


Windows PowerShell:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
```

### 3. Run demo
```
python scripts/run_demo.py --write-logs
```

### 4. Open notebooks  
Follow the order:  
`01 → 02 → 03 → 04`

---

# Running Tests

```
pytest -q
```
---

## Blog ↔ Repo Mapping (Complete Reference)

This mapping ensures readers understand exactly where each concept is implemented.

| Blog Section | Repo Module | Notes |
|-------------|-------------|-------|
| **Instrumentation (Intrinsic Metrics)** | `metrics.py` → `compute_intrinsic_metrics` | Same logic as blog; repo adds validation + docstring |
| **Instrumentation (Extrinsic Metrics)** | `metrics.py` → `compute_extrinsic_metrics` | Tracks latency, tokens, retrieval time |
| **Instrumentation (Behavioral Metrics)** | `metrics.py` → `compute_behavioral_metrics` | Supports verbosity; easy to extend |
| **Combined Metrics Example** | `metrics.py` → `compute_all_metrics` | Wraps all metric families in one object |
| **Hands-On Code Snippet** | `notebooks/01_metrics_and_signals.ipynb` | Recreates + extends blog example |
| **Stage A / Stage B / Stage C** |  `stages.py` | Will implement multi-stage evaluation |
| **Evaluation Controller** |  `controller.py` | Will convert signals into corrective action |
| **Feedback Loops** | `feedback_loops.py` | Will simulate offline & online learning |
| **Engineering Patterns** |  `patterns.py` | Canary prompts, audits, shadow evaluation |

This table will evolve as future blog articles extend the project.

---

## Differences from the Blog Code Snippets

To keep the article readable, the blog uses minimal Python examples.

The repo version:

- includes type hints  
- validates input data  
- uses a dataclass (`MetricsResult`) to hold results  
- supports optional retrieval latency  
- exposes a CLI runnable module (`python -m rag_eval.metrics`)  
- provides a notebook with richer examples  

Conceptually, the logic remains identical.  
Practically, the repo is structured for real engineering use.

---
