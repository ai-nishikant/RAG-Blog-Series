"""
Retrieval-Aware Prompting Demo

This script demonstrates Retrieval-Aware Prompting in RAG systems.
Retrieval-aware prompting doesn't treat retrieved data as static text. Instead,
it tailors instructions dynamically based on metadata such as recency or confidence scores.

Why It Matters: Ensures context is prioritized intelligently — using the most relevant
or recent data first, which is crucial for time-sensitive or multi-domain information.

Example: "Use the context passages with the highest relevance score first."

Applied Use Case: News summarization RAG system that prioritizes recent articles
over older ones, keeping summaries aligned with latest events in domains like
finance or cybersecurity.

This demo compares:
- Baseline: Standard retrieval (unsorted context)
- Recency-weighted: Context sorted by recency using filename date prefixes
"""

from pathlib import Path
import yaml
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, sort_by_recency_filenames, Timer

# Get the project root directory (parent of scripts folder)
HERE = Path(__file__).resolve().parent.parent

# Load prompt templates from YAML configuration
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_retrieval_aware(context: str) -> str:
    """
    Create a retrieval-aware prompt that leverages context ordering.

    This function generates prompts that instruct the model to prioritize
    information based on context ordering. The prompt template includes
    instructions like "Use the most recent and relevant context first"
    to ensure intelligent prioritization of retrieved data.

    Args:
        context (str): The retrieved and ordered context (can be sorted by recency, relevance, etc.)

    Returns:
        str: The formatted retrieval-aware prompt ready for the language model
    """
    # Get the retrieval-aware template that includes prioritization instructions
    tpl = TEMPLATES["templates"]["retrieval_aware"]

    # Combine opening instruction with the formatted body containing context ordering guidance
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context)

def main():
    """
    Main function demonstrating retrieval-aware prompting with recency-based prioritization.

    This function compares two approaches to context handling in RAG:
    1. Baseline: Standard retrieval (unsorted context) - treats data as static text
    2. Recency-weighted: Context sorted by recency - prioritizes recent information first

    The comparison shows how retrieval-aware prompting can improve results in
    time-sensitive domains like news summarization by leveraging metadata (recency).
    """
    # Initialize the language model client for our retrieval-aware demo
    client = GroqClient()

    # Define the user question for news summarization
    # This represents a typical query in time-sensitive information systems
    question = "Summarize the latest developments in 3-4 sentences."

    # Print the user question
    print("User Question:")
    print(question)
    print("\n" + "="*60 + "\n")

    # === BASELINE APPROACH ===
    # Load news articles without considering recency metadata
    # This simulates traditional RAG that treats retrieved data as static text
    baseline = "\n\n".join(read_texts_from_dir(str(HERE / "data" / "news"), limit=3))

    print("=== Baseline Approach (unsorted context) ===")
    print("Context treated as static text - no prioritization based on recency")
    baseline_prompt = render_retrieval_aware(baseline)
    print("\n--- Baseline Prompt ---\n", baseline_prompt)
    print("\n--- Baseline Response ---\n", end="")
    with Timer() as tb:
        out_base = client.generate(baseline_prompt)
    print(out_base, "\nLatency:", round(tb.elapsed, 2), "s")

    # === RECENCY-WEIGHTED APPROACH ===
    # Load and sort news articles by recency using filename date prefixes
    # This demonstrates retrieval-aware prompting that leverages metadata for intelligent prioritization
    # Most recent articles appear first in the context, allowing the model to prioritize recent events
    sorted_ctx = "\n\n".join(sort_by_recency_filenames(str(HERE / "data" / "news"), limit=3))

    print("\n" + "="*60)
    print("=== Recency-weighted Approach (sorted context) ===")
    print("Context ordered by recency - recent articles prioritized for time-sensitive summarization")
    sorted_prompt = render_retrieval_aware(sorted_ctx)
    print("\n--- Recency-weighted Prompt ---\n", sorted_prompt)
    print("\n--- Recency-weighted Response ---\n", end="")
    with Timer() as ts:
        out_sorted = client.generate(sorted_prompt)
    print(out_sorted, "\nLatency:", round(ts.elapsed, 2), "s")

if __name__ == "__main__":
    # Execute the Retrieval-Aware Prompting demo when script is run directly
    # This allows the script to be imported without running the demo
    main()
