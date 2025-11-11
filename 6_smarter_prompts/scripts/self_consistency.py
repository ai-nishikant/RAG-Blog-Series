import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer
from src.eval_metrics import rouge_scores

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_sc(context: str, question: str) -> str:
    tpl = TEMPLATES["templates"]["self_consistency"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def consensus(outputs: list[str]) -> str:
    # Naive consensus: choose the answer with highest average ROUGE against others.
    best, best_score = outputs[0], -1.0
    for i, oi in enumerate(outputs):
        scores = []
        for j, oj in enumerate(outputs):
            if i == j: 
                continue
            scores.append(rouge_scores(oj, oi)["rouge1"])
        avg = sum(scores) / len(scores) if scores else 0
        if avg > best_score:
            best, best_score = oi, avg
    return best

def main():
    client = GroqClient()
    profiles = read_texts_from_dir(str(HERE / "data" / "vendor_risk"), limit=3)
    context = "\n\n".join(profiles)
    question = "Which vendor appears lower overall risk and why, in one short sentence?"

    outputs = []
    with Timer() as t:
        for _ in range(3):
            outputs.append(client.generate(render_sc(context, question), temperature=0.8))
    print("=== Candidates ===")
    for k, o in enumerate(outputs, 1):
        print(f"\n[{k}] {o}")

    print("\n=== Consensus (naive ROUGE-based) ===")
    print(consensus(outputs))
    print("\nTotal latency:", round(t.elapsed, 2), "s")

if __name__ == "__main__":
    main()
