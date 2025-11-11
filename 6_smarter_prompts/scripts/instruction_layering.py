import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_stage1(context: str) -> str:
    tpl = TEMPLATES["templates"]["instruction_layering_stage1"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context)

def render_stage2(summary: str, question: str) -> str:
    tpl = TEMPLATES["templates"]["instruction_layering_stage2"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(summary=summary, question=question)

def main():
    client = GroqClient()
    clauses = read_texts_from_dir(str(HERE / "data" / "contracts"), limit=2)
    context = "\n\n".join(clauses)
    question = "State the incident notification window and data deletion commitment."

    with Timer() as t1:
        summary = client.generate(render_stage1(context))
    with Timer() as t2:
        final = client.generate(render_stage2(summary, question), max_tokens=200)

    print("=== Stage 1: Summary ===\n", summary)
    print("\n=== Stage 2: Final Answer ===\n", final)
    print("\nLatency — Stage1:", round(t1.elapsed, 2), "s  Stage2:", round(t2.elapsed, 2), "s")

if __name__ == "__main__":
    main()
