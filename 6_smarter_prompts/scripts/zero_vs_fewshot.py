"""
Zero-shot vs Few-shot Prompting Comparison Demo

This script demonstrates the difference between zero-shot and few-shot prompting
strategies in RAG (Retrieval-Augmented Generation) systems. It compares how
a language model performs when given just instructions (zero-shot) versus when
provided with examples of the desired behavior (few-shot).

The demo uses customer support ticket scenarios to show how few-shot examples
can improve response quality and consistency.
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

# Load few-shot examples from YAML configuration
FEWSHOT = yaml.safe_load((HERE / "prompts" / "few_shot_examples.yaml").read_text())

def render_zero_shot(context: str, question: str) -> str:
    """
    Create a zero-shot prompt with just instructions and context.
    
    Zero-shot prompting relies solely on the model's pre-trained knowledge
    and the instructions provided, without any examples of the desired output format.
    
    Args:
        context (str): The retrieved documents/context for the query
        question (str): The user's question to be answered
        
    Returns:
        str: The formatted prompt ready for the language model
    """
    # Get the zero-shot template from our configuration
    tpl = TEMPLATES["templates"]["zero_shot"]
    
    # Combine opening instruction with the formatted body
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context, question=question)

def render_few_shot(context: str, question: str) -> str:
    """
    Create a few-shot prompt with examples of desired behavior.
    
    Few-shot prompting provides the model with examples of input-output pairs
    to demonstrate the expected response format and style. This helps the model
    understand the task better and produce more consistent outputs.
    
    Args:
        context (str): The retrieved documents/context for the query
        question (str): The user's question to be answered
        
    Returns:
        str: The formatted prompt with examples, ready for the language model
    """
    # Extract the few-shot examples from our configuration
    examples = FEWSHOT["support_examples"]
    
    # Format each example into a consistent structure
    ex_lines = []
    for ex in examples:
        # Each example shows the model: Context -> Question -> Expected Answer
        ex_lines.append(f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n")
    
    # Join all examples into a single string
    few = "\n".join(ex_lines)
    
    # Get the few-shot template from our configuration
    tpl = TEMPLATES["templates"]["few_shot"]
    
    # Format the template with examples, context, and question
    body = tpl["body"].format(few_shot_examples=few, context=context, question=question)
    
    # Combine opening instruction with the formatted body
    return f"{tpl['opening_instruction']}\n\n{body}"

def main():
    """
    Main function that demonstrates zero-shot vs few-shot prompting comparison.
    
    This function:
    1. Loads customer support tickets as context
    2. Defines a test question about double charges
    3. Generates responses using both prompting strategies
    4. Measures and compares the performance using ROUGE metrics
    """
    # Initialize the language model client (using Groq in this case)
    client = GroqClient()
    
    # Load sample support tickets to use as context for our RAG system
    # Limiting to 3 tickets to keep the context manageable
    tickets = read_texts_from_dir(str(HERE / "data" / "support_tickets"), limit=3)
    
    # Combine all tickets into a single context string
    # This simulates the retrieved documents in a real RAG pipeline
    context = "\n\n".join(tickets)
    
    # Define our test question - a common customer support scenario
    question = "A customer reports a double charge and wants a refund. What should we reply?"


    # Print the user question
    print("\nUser Question:\n", question)

    # === ZERO-SHOT PROMPTING ===
    zero_shot_prompt = render_zero_shot(context, question)
    print("\n--- Zero-shot Prompt ---\n", zero_shot_prompt)
    with Timer() as t0:
        out0 = client.generate(zero_shot_prompt)
    print("\n--- Zero-shot Response ---\n", out0)
    print("Latency:", round(t0.elapsed, 2), "s")

    # === FEW-SHOT PROMPTING ===
    few_shot_prompt = render_few_shot(context, question)
    print("\n--- Few-shot Prompt ---\n", few_shot_prompt)
    with Timer() as t1:
        out1 = client.generate(few_shot_prompt)
    print("\n--- Few-shot Response ---\n", out1)
    print("Latency:", round(t1.elapsed, 2), "s")

    # === EVALUATION ===
    # Compare both approaches against a reference answer using ROUGE metrics
    # Note: This is a simplified reference for demonstration purposes
    # In production, you'd want multiple reference answers and human evaluation
    reference = "Apologize for the duplicate charge, issue a refund for the extra billing, and send a confirmation email."
    
    # ROUGE scores measure overlap between generated text and reference
    # Higher scores indicate better similarity to the expected response
    print("\nROUGE vs reference (Zero-shot):", rouge_scores(reference, out0))
    print("ROUGE vs reference (Few-shot):", rouge_scores(reference, out1))

if __name__ == "__main__":
    # Execute the comparison demo when script is run directly
    # This allows the script to be imported without running the demo
    main()
