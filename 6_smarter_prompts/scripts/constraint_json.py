"""
Constraint-based Prompting Demo

This script demonstrates Constraint-based Prompting in RAG systems.
Constraint-based prompting enforces structured output formats and strict adherence
to provided data, ensuring machine-readable and verifiable outputs.

Why It Matters: Guarantees consistent, structured responses that can be reliably
parsed and processed by downstream systems, eliminating ambiguity in critical applications.

When to Use: Compliance summaries, financial reports, structured analytics, or any
scenario requiring predictable, machine-readable outputs.

Example: "Answer in JSON format with fields: policy_name, year, key_changes."

Applied Use Case: Corporate knowledge base where all responses follow a defined
JSON schema, enabling downstream automation and maintaining audit compliance.
This ensures structured outputs for compliance summaries that can be automatically
processed and verified.
"""

import json
from jsonschema import validate, ValidationError
from pathlib import Path
import yaml
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer

# Get the project root directory (parent of scripts folder)
HERE = Path(__file__).resolve().parent.parent

# Load prompt templates from YAML configuration
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

# Define JSON schema for structured output validation
# This schema enforces the exact structure required for compliance summaries
SCHEMA = {
    "type": "object",
    "properties": {
        "policy_name": {"type": "string"},
        "year": {"type": "integer"},
        "key_changes": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["policy_name", "year", "key_changes"],
    "additionalProperties": False
}

def render_constraint(context: str, question: str) -> str:
    """
    Create a constraint-based prompt that enforces structured JSON output.

    This function generates prompts that require the model to produce machine-readable,
    verifiable outputs following strict formatting rules. The prompt includes specific
    instructions like "Answer in JSON format with fields: policy_name, year, key_changes"
    to ensure structured compliance with the defined schema.

    Args:
        context (str): The compliance policy documents to extract structured data from
        question (str): The specific query requiring structured output (e.g., JSON format)

    Returns:
        str: The formatted constraint-based prompt that enforces structured output
    """
    # Get the constraint-based template that includes JSON formatting requirements
    tpl = TEMPLATES["templates"]["constraint_json"]

    # Combine opening instruction with formatted body containing structure enforcement
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def try_parse_json(text: str) -> dict | None:
    """
    Attempt to parse JSON from model output for constraint validation.

    This function extracts and validates JSON structure from the model's response,
    which is critical for constraint-based prompting. It ensures the output is
    machine-readable and can be processed by downstream automation systems.

    In constraint-based prompting, successful parsing confirms the model adhered
    to the required structured format, enabling reliable data extraction for
    compliance summaries and automated processing.

    Args:
        text (str): The raw model output containing potential JSON structure

    Returns:
        dict | None: Parsed JSON object if valid, None if parsing fails
    """
    # Extract first JSON object-like block heuristically
    # This handles cases where the model includes extra text around the JSON
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
    """
    Main function demonstrating constraint-based prompting for structured compliance outputs.

    This function shows how constraint-based prompting ensures machine-readable outputs
    for compliance summaries. It enforces strict adherence to JSON schema requirements,
    enabling downstream automation and maintaining audit compliance in corporate knowledge bases.

    The process validates that outputs follow the exact structure needed for:
    - Automated processing systems
    - Compliance reporting
    - Structured analytics
    - Audit trails
    """
    # Initialize the language model client for constraint-based generation
    client = GroqClient()

    # Load compliance policy documents for structured extraction
    # This represents the type of data used in corporate knowledge bases
    policies = read_texts_from_dir(str(HERE / "data" / "compliance_policies"), limit=1)
    context = "\n\n".join(policies)

    # Define the question requiring structured JSON output
    # This exemplifies the type of query used in compliance and financial reporting
    question = "Return the policy name, year, and key changes as JSON."

    # Print the user question for clarity
    print("User Question:")
    print(question)
    print("\n" + "="*60 + "\n")

    # Generate constraint-based response with JSON format enforcement
    # The prompt includes specific instructions for structured output
    constraint_prompt = render_constraint(context, question)
    print("Constraint-based Prompt:")
    print(constraint_prompt)
    print("\n" + "="*60 + "\n")

    with Timer() as t:
        output = client.generate(constraint_prompt, max_tokens=200)

    # Display the raw model output
    print("Raw Model Output:")
    print(output)
    print("\n" + "="*60 + "\n")

    # === CONSTRAINT VALIDATION ===
    # Parse the JSON response - critical for machine-readability
    obj = try_parse_json(output)
    if obj is None:
        print("Structured Output: INVALID (could not parse JSON)")
        print("❌ Constraint-based prompting failed - output not machine-readable")
        return

    # Validate against JSON schema - ensures compliance with required structure
    try:
        validate(instance=obj, schema=SCHEMA)
        print("Structured Output: VALID ✅")
        print("✅ Constraint-based prompting succeeded - output is machine-readable and schema-compliant")
        print("\nValidated JSON Structure:")
        print(json.dumps(obj, indent=2))
        print("\n📋 This structured output enables:")
        print("   • Downstream automation systems")
        print("   • Automated compliance reporting")
        print("   • Structured analytics processing")
        print("   • Audit trail maintenance")
    except ValidationError as e:
        print("Structured Output: INVALID ❌")
        print("❌ Schema validation failed:", e.message)
        print("💡 Constraint-based prompting ensures outputs meet exact structural requirements")

    print(f"\n⏱️  Processing Latency: {round(t.elapsed, 2)}s")

if __name__ == "__main__":
    # Execute the Constraint-based Prompting demo when script is run directly
    # This allows the script to be imported without running the demo
    main()
