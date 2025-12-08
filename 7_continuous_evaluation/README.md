# Continuous Evaluation for RAG Systems  
Companion Code for the Blog:  
**“Why RAG Systems Drift — And How Continuous Evaluation Keeps Them Stable”**

This repository contains the companion code for the seventh article in the RAG Blog Series.  
It implements the foundations of a **continuous evaluation pipeline** for RAG systems, focusing on:

- capturing instrumentation signals
- measuring intrinsic / extrinsic / behavioral metrics
- preparing these signals for multi-stage evaluation
- building toward controllers and feedback loops discussed in the blog

This repo is deliberately structured so readers can run, extend, and adapt the ideas from the article in a real project environment.

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
| Step 2 — Multi-Stage Evaluation Loop | Localize failures (A/B/C) | *(to be added)* |
| Step 3 — Evaluation Controller | Real-time corrective actions | *(to be added)* |
| Step 4 — Feedback Loops | Offline/online continuous improvement | *(to be added)* |
| Step 5 — Engineering Patterns | Canary prompts, shadow eval, audits | *(to be added)* |
| Hands-On Example | Minimal runnable demo | `notebooks/01_metrics_and_signals.ipynb` |

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
├── src/
│   └── rag_eval/
│       ├── __init__.py
│       └── metrics.py
├── notebooks/
│   └── 01_metrics_and_signals.ipynb
└── data/
    └── (optional future synthetic examples)
```

Future expansions (not required to run Step 1):

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

### Install dependencies

```
pip install -r requirements.txt
```

A minimal `requirements.txt` for Step 1:

```
numpy>=1.24
pandas>=2.0
```

### Make `src/` importable

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
```

---

## Quickstart: Run the Demo

Once installed:

```bash
python -m rag_eval.metrics
```

This command:

- computes intrinsic, extrinsic, and behavioral metrics  
- prints signals used later in multi-stage evaluation and controllers  

### Run the notebook:

```bash
jupyter notebook notebooks/01_metrics_and_signals.ipynb
```

The notebook:

- reproduces the blog’s code example  
- evaluates multiple reference/output pairs  
- demonstrates how to assemble metrics into usable evaluation features  

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
| **Stage A / Stage B / Stage C** | *(future)* `stages.py` | Will implement multi-stage evaluation |
| **Evaluation Controller** | *(future)* `controller.py` | Will convert signals into corrective action |
| **Feedback Loops** | *(future)* `feedback_loops.py` | Will simulate offline & online learning |
| **Engineering Patterns** | *(future)* `patterns.py` | Canary prompts, audits, shadow evaluation |

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

## Contributing

This folder will expand as subsequent blog posts introduce:

- multi-stage evaluation  
- controller logic  
- drift detection  
- dashboards  
- adaptive prompt strategies  

Contributions are welcome.

