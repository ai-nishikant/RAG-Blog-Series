import json
from jsonschema import validate, ValidationError
from pathlib import Path
import yaml
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer

HERE = Path(__file__).resolve().parent.parent
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

SCHEMA = {
    "type": "object",
    "properties": {
        "policy_name": {"type": "string"},
        "year": {"type": "string"},
        "key_changes": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["policy_name", "year", "key_changes"],
    "additionalProperties": False
}

def render_constraint(context: str, question: str) -> str:
    tpl = TEMPLATES["templates"]["constraint_json"]
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def try_parse_json(text: str) -> dict | None:
    # Extract first JSON object-like block heuristically
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start: end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

def main():
    client = GroqClient()
    policies = read_texts_from_dir(str(HERE / "data" / "compliance_policies"), limit=1)
    context = "\n\n".join(policies)
    question = "Return the policy name, year, and key changes as JSON."

    with Timer() as t:
        output = client.generate(render_constraint(context, question), max_tokens=200)
    print("=== Raw Model Output ===\n", output, "\n")

    obj = try_parse_json(output)
    if obj is None:
        print("Parsed JSON: INVALID (could not parse)")
        return

    try:
        validate(instance=obj, schema=SCHEMA)
        print("Parsed JSON: VALID")
        print(json.dumps(obj, indent=2))
    except ValidationError as e:
        print("Parsed JSON: INVALID")
        print("Validation error:", e.message)

    print("\nLatency:", round(t.elapsed, 2), "s")

if __name__ == "__main__":
    main()
