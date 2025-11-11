import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer
from src.eval_metrics import rouge_scores

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())
FEWSHOT = yaml.safe_load((HERE / "prompts" / "few_shot_examples.yaml").read_text())

def render_zero_shot(context: str, question: str) -> str:
    tpl = TEMPLATES["templates"]["zero_shot"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def render_few_shot(context: str, question: str) -> str:
    examples = FEWSHOT["support_examples"]
    ex_lines = []
    for ex in examples:
        ex_lines.append(f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n")
    few = "\n".join(ex_lines)
    tpl = TEMPLATES["templates"]["few_shot"]
    body = tpl["body"].format(few_shot_examples=few, context=context, question=question)
    return f"{tpl['opening_instruction']}\n\n{body}"

def main():
    client = GroqClient()
    tickets = read_texts_from_dir(str(HERE / "data" / "support_tickets"), limit=3)
    context = "\n\n".join(tickets)
    question = "A customer reports a double charge and wants a refund. What should we reply?"

    # Zero-shot
    with Timer() as t0:
        out0 = client.generate(render_zero_shot(context, question))
    print("=== Zero-shot ===\n", out0, "\nLatency:", round(t0.elapsed, 2), "s")

    # Few-shot
    with Timer() as t1:
        out1 = client.generate(render_few_shot(context, question))
    print("\n=== Few-shot ===\n", out1, "\nLatency:", round(t1.elapsed, 2), "s")

    # Optional quick metric (tiny reference for demonstration only)
    reference = "Apologize for the duplicate charge, issue a refund for the extra billing, and send a confirmation email."
    print("\nROUGE vs reference (Zero-shot):", rouge_scores(reference, out0))
    print("ROUGE vs reference (Few-shot):", rouge_scores(reference, out1))

if __name__ == "__main__":
    main()
