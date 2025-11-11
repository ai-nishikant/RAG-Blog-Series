"""
Instruction Layering Prompting Demo

This script demonstrates Instruction Layering prompting in RAG systems.
Instruction Layering separates global rules from task-specific directives to promote
modularity and consistent behavior across tasks.

Global Rules (applied across all stages):
- Always base answers on verified context
- Maintain factual accuracy and consistency

Task-Specific Directives (stage-dependent):
- Stage 1: "Summarize this report in one paragraph"
- Stage 2: "Answer questions clearly and briefly using only the summary"

Why It Matters: Multi-stage RAG workflows become predictable and easier to orchestrate at scale.

Applied Use Case: Contract analysis pipeline where multiple LLM components collaborate:
- Component 1: Summarizes contract clauses
- Component 2: Extracts specific clauses
- Component 3: Generates executive insights
Global rules ensure every step stays consistent and factual.
"""

import yaml
from pathlib import Path
from src.llm_client import GroqClient
from src.utils import read_texts_from_dir, Timer

# Get the project root directory (parent of scripts folder)
HERE = Path(__file__).resolve().parent.parent

# Load prompt templates from YAML configuration
TEMPLATES = yaml.safe_load((HERE / "prompts" / "templates.yaml").read_text())

def render_stage1(context: str) -> str:
    """
    Create the first stage prompt with task-specific summarization directives.

    Stage 1 applies global rules (factual accuracy, context-based answers) combined with
    task-specific directives for summarization: "Summarize key risks and obligations in one short paragraph".

    This stage focuses on creating a concise summary that other pipeline components
    can reliably use as input, following the modular approach of instruction layering.

    Args:
        context (str): The contract clauses to be summarized

    Returns:
        str: The formatted Stage 1 prompt with layered instructions
    """
    # Get the Stage 1 template with task-specific summarization directives
    tpl = TEMPLATES["templates"]["instruction_layering_stage1"]

    # Combine global rules with task-specific summarization instructions
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(context=context)

def render_stage2(summary: str, question: str) -> str:
    """
    Create the second stage prompt with task-specific question-answering directives.

    Stage 2 applies the same global rules (factual accuracy, context-based answers) but with
    different task-specific directives: "Answer questions clearly and briefly using only the summary".

    This demonstrates how instruction layering enables modular RAG pipelines where each
    component has specialized instructions while maintaining consistent global behavior.

    Args:
        summary (str): The summary generated in Stage 1 (output from previous pipeline component)
        question (str): The specific question requiring precise information extraction

    Returns:
        str: The formatted Stage 2 prompt with layered instructions
    """
    # Get the Stage 2 template with task-specific question-answering directives
    tpl = TEMPLATES["templates"]["instruction_layering_stage2"]

    # Combine global rules with task-specific question-answering instructions
    return f"{tpl['opening_instruction']}\n\n" + tpl["body"].format(summary=summary, question=question)

def main():
    """
    Main function demonstrating Instruction Layering in a multi-stage RAG workflow.

    This function implements a two-stage contract analysis pipeline:
    - Stage 1 (Summarization Component): Applies global rules + summarization directives
    - Stage 2 (Question-Answering Component): Applies same global rules + QA directives

    This demonstrates how instruction layering enables predictable, modular RAG pipelines
    where multiple LLM components can collaborate while maintaining consistent behavior.
    """
    # Initialize the language model client for our multi-stage pipeline
    client = GroqClient()

    # Load contract clauses as input for the first pipeline component
    # In a real pipeline, this would be retrieved documents from a vector database
    clauses = read_texts_from_dir(str(HERE / "data" / "contracts"), limit=2)
    context = "\n\n".join(clauses)

    # Define the question for the final pipeline component
    question = "State the incident notification window and data deletion commitment."

    # === STAGE 1: SUMMARIZATION COMPONENT ===
    # Global rules: Always base answers on verified context
    # Task-specific: Summarize key risks and obligations in one short paragraph
    # This component prepares condensed input for downstream components
    with Timer() as t1:
        summary = client.generate(render_stage1(context))

    # === STAGE 2: QUESTION-ANSWERING COMPONENT ===
    # Global rules: Always base answers on verified context (same as Stage 1)
    # Task-specific: Answer questions clearly and briefly using only the summary
    # This component uses Stage 1 output to provide precise, factual answers
    with Timer() as t2:
        final = client.generate(render_stage2(summary, question), max_tokens=200)

    # Display the complete pipeline execution
    print("=== Stage 1: Summarization Component ===\n", summary)
    print("\n=== Stage 2: Question-Answering Component ===\n", final)

    # Show timing for pipeline orchestration
    print("\nLatency — Summarization:", round(t1.elapsed, 2), "s  QA:", round(t2.elapsed, 2), "s")

if __name__ == "__main__":
    # Execute the Instruction Layering demo when script is run directly
    # This allows the script to be imported without running the demo
    main()
