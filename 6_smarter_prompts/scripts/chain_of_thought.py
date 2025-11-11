import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_cot(context: str, question: str) -> str:
    tpl = TEMPLATES["templates"]["chain_of_thought"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def main():
    client = GroqClient()
    policies = read_texts_from_dir(str(HERE / "data" / "compliance_policies"), limit=2)
    context = "\n\n".join(policies)
    question = "Summarize the new auditing requirements and any vendor obligations."

    with Timer() as t:
        output = client.generate(render_cot(context, question), max_tokens=600)
    print("=== Chain-of-Thought ===\n", output)
    print("\nLatency:", round(t.elapsed, 2), "s")

if __name__ == "__main__":
    main()
