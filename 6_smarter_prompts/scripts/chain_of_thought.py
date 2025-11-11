"""
Chain-of-Thought Prompting Demo

This script demonstrates Chain-of-Thought (CoT) prompting in RAG systems.
Chain-of-Thought prompting encourages language models to reason step-by-step
before providing a final answer, leading to more accurate and well-reasoned responses.

The demo uses compliance policy documents to show how CoT prompting can help
analyze complex regulatory requirements and extract key obligations systematically.
"""

import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer

# Get the project root directory (parent of scripts folder)
HERE = Path(__file__).resolve().parent.parent

# Load prompt templates from YAML configuration
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_cot(context: str, question: str) -> str:
    """
    Create a Chain-of-Thought prompt that encourages step-by-step reasoning.

    Chain-of-Thought prompting asks the model to break down complex problems
    into intermediate reasoning steps before arriving at a final answer.
    This approach often leads to more accurate and explainable responses.

    Args:
        context (str): The retrieved documents/context for the query
        question (str): The user's question requiring analytical reasoning

    Returns:
        str: The formatted Chain-of-Thought prompt ready for the language model
    """
    # Get the Chain-of-Thought template from our configuration
    tpl = TEMPLATES["templates"]["chain_of_thought"]

    # Combine opening instruction with the formatted body
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def main():
    """
    Main function that demonstrates Chain-of-Thought prompting.

    This function:
    1. Loads compliance policy documents as context
    2. Defines a question requiring analytical reasoning
    3. Generates a response using Chain-of-Thought prompting
    4. Measures and displays the response time
    """
    # Initialize the language model client (using Groq in this case)
    client = GroqClient()

    # Load sample compliance policy documents to use as context
    # Limiting to 2 documents to keep the context focused
    policies = read_texts_from_dir(str(HERE / "data" / "compliance_policies"), limit=2)

    # Combine all policy documents into a single context string
    # This simulates the retrieved documents in a real RAG pipeline
    context = "\n\n".join(policies)

    # Define our test question that requires analytical reasoning
    # This question needs the model to identify and summarize multiple requirements
    question = "Summarize the new auditing requirements and any vendor obligations."

    # Print the user question
    print("\nUser Question:\n", question)

    # Generate response using Chain-of-Thought prompting
    # Increased max_tokens to allow for detailed step-by-step reasoning
    with Timer() as t:
        cot_prompt = render_cot(context, question)
        print("\nn=== Chain-of-Thought Prompt ===\n", cot_prompt)
        output = client.generate(cot_prompt, max_tokens=600)

    # Display the Chain-of-Thought response
    print("\n\n=== Chain-of-Thought Response ===\n", output)
    print("\nLatency:", round(t.elapsed, 2), "s")

if __name__ == "__main__":
    # Execute the Chain-of-Thought demo when script is run directly
    # This allows the script to be imported without running the demo
    main()
