from pathlib import Path
import yaml
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, sort_by_recency_filenames, Timer

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_retrieval_aware(context: str) -> str:
    tpl = TEMPLATES["templates"]["retrieval_aware"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context)

def main():
    client = GroqClient()
    # Baseline: unsorted
    baseline = "\n\n".join(read_texts_from_dir(str(HERE / "data" / "news"), limit=3))
    # Recency-weighted: sorted by filename date prefix
    sorted_ctx = "\n\n".join(sort_by_recency_filenames(str(HERE / "data" / "news"), limit=3))

    print("=== Baseline (unsorted) ===")
    with Timer() as tb:
        out_base = client.generate(render_retrieval_aware(baseline))
    print(out_base, "\nLatency:", round(tb.elapsed, 2), "s")

    print("\n=== Recency-weighted (sorted) ===")
    with Timer() as ts:
        out_sorted = client.generate(render_retrieval_aware(sorted_ctx))
    print(out_sorted, "\nLatency:", round(ts.elapsed, 2), "s")

if __name__ == "__main__":
    main()
