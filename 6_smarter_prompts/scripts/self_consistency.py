"""
Self-Consistency Prompting Demo

This script demonstrates Self-Consistency prompting in RAG systems.
Self-Consistency prompting generates multiple independent responses to the same query
and then selects the most consistent answer among them. This approach often leads to
more reliable and accurate responses by leveraging the model's natural variations.

The demo uses vendor risk assessment data to show how self-consistency can help
determine the most reliable answer when evaluating complex risk factors.
"""

import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer
from src.eval_metrics import rouge_scores

# Get the project root directory (parent of scripts folder)
HERE = Path(__file__).resolve().parent.parent

# Load prompt templates from YAML configuration
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_sc(context: str, question: str) -> str:
    """
    Create a Self-Consistency prompt for generating independent responses.

    Self-Consistency prompting uses the same prompt template as other methods
    but relies on generating multiple responses with some randomness (via temperature)
    to explore different reasoning paths, then selecting the most consistent answer.

    Args:
        context (str): The retrieved documents/context for the query
        question (str): The user's question requiring evaluation

    Returns:
        str: The formatted prompt ready for the language model
    """
    # Get the self-consistency template from our configuration
    tpl = TEMPLATES["templates"]["self_consistency"]

    # Combine opening instruction with the formatted body
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def consensus(outputs: list[str]) -> str:
    """
    Determine the most consistent answer using a naive ROUGE-based consensus method.

    This function implements a simple consensus algorithm that selects the answer
    with the highest average ROUGE similarity score against all other answers.
    The idea is that the most "consistent" answer will be the one that best
    agrees with the other generated responses.

    Args:
        outputs (list[str]): List of generated responses to compare

    Returns:
        str: The response with highest consensus score
    """
    # Initialize with first output as baseline
    best, best_score = outputs[0], -1.0

    # Compare each output against all others
    for i, oi in enumerate(outputs):
        scores = []
        # Calculate ROUGE similarity against every other output
        for j, oj in enumerate(outputs):
            if i == j:
                continue  # Skip comparing output against itself
            # Use ROUGE-1 score as similarity measure
            scores.append(rouge_scores(oj, oi)["rouge1"])

        # Calculate average similarity score
        avg = sum(scores) / len(scores) if scores else 0

        # Update best answer if this one has higher consensus
        if avg > best_score:
            best, best_score = oi, avg

    return best

def main():
    """
    Main function that demonstrates Self-Consistency prompting.

    This function:
    1. Loads vendor risk assessment documents as context
    2. Generates multiple independent responses using temperature variation
    3. Applies consensus algorithm to select the most consistent answer
    4. Displays all candidates and the final consensus choice
    """
    # Initialize the language model client (using Groq in this case)
    client = GroqClient()

    # Load sample vendor risk assessment documents
    # Limiting to 3 documents to keep the context manageable
    profiles = read_texts_from_dir(str(HERE / "data" / "vendor_risk"), limit=3)

    # Combine all vendor profiles into a single context string
    # This simulates the retrieved documents in a real RAG pipeline
    context = "\n\n".join(profiles)

    # Define our test question requiring risk assessment and reasoning
    question = "Which vendor appears lower overall risk and why, in one short sentence?"

    # Generate multiple independent responses
    outputs = []
    with Timer() as t:
        # Generate 3 responses with temperature=0.8 for some variation
        # Higher temperature introduces controlled randomness for diversity
        for _ in range(3):
            consistency_prompt = render_sc(context, question)
            print("\n=== Self-Consistency Prompt ===\n", consistency_prompt)
            outputs.append(client.generate(consistency_prompt, temperature=0.8))

    # Display all candidate responses
    print("=== Candidates ===")
    for k, o in enumerate(outputs, 1):
        print(f"\n[{k}] {o}")

    # Apply consensus algorithm to find the most consistent answer
    print("\n=== Consensus (naive ROUGE-based) ===")
    print(consensus(outputs))

    # Show total time for all generations
    print("\nTotal latency:", round(t.elapsed, 2), "s")

if __name__ == "__main__":
    # Execute the Self-Consistency demo when script is run directly
    # This allows the script to be imported without running the demo
    main()
