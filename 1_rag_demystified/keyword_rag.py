"""
Toy Keyword-Based Q&A System
---------------------------------
This example shows why retrieval matters.
We simulate a knowledge base with plain text,
perform keyword search, and let the LLM-like
function answer using retrieved context.
"""

# Import required modules
import re
from typing import List, Tuple

# Simulated knowledge base (a few docs)
# Each document is a simple string mapped to a doc ID
DOCUMENTS = {
    "doc1": "RAG stands for Retrieval-Augmented Generation. It helps LLMs fetch relevant context before answering.",
    "doc2": "Large Language Models (LLMs) are powerful but prone to hallucinations. They sometimes make up facts.",
    "doc3": "In enterprise settings, RAG reduces compliance risk by grounding answers in real company knowledge."
}


def retrieve(query: str, documents: dict, top_k: int = 1) -> List[Tuple[str, str]]:
    """
    Retrieve documents matching the query using simple keyword search.
    Args:
        query: The user's question or search string.
        documents: Dictionary of doc_id to content.
        top_k: Number of top documents to return.
    Returns:
        List of (doc_id, content) tuples for top matches.
    """
    # Split query into lowercase terms
    query_terms = query.lower().split()
    scores = []
    
    # Score each document by number of matching terms
    for doc_id, content in documents.items():
        # Count how many query terms appear in the document
        matches = sum(1 for term in query_terms if re.search(rf"\b{term}\b", content.lower()))
        scores.append((doc_id, content, matches))
    
    # Sort documents by score (number of matches), descending
    ranked = sorted(scores, key=lambda x: x[2], reverse=True)
    # Return top_k documents (doc_id, content)
    return [(doc_id, content) for doc_id, content, _ in ranked[:top_k]]


def generate_answer(query: str, retrieved_docs: List[Tuple[str, str]]) -> str:
    """
    Generate an answer based on retrieved context.
    (Simulated LLM -- here we just concatenate info.)
    Args:
        query: The user's question.
        retrieved_docs: List of (doc_id, content) tuples.
    Returns:
        A string answer using the retrieved context.
    """
    if not retrieved_docs:
        # No relevant docs found
        return "Sorry, I could not find relevant information."
    
    # Concatenate all retrieved document contents
    context = " ".join([doc for _, doc in retrieved_docs])
    return f"Answering based on retrieved context: {context}"


# --- Example Run ---
if __name__ == "__main__":
    # Example question to ask the system
    question = "What is RAG?"
    # Retrieve top 2 relevant documents
    docs = retrieve(question, DOCUMENTS, top_k=2)
    # Generate answer using retrieved docs
    answer = generate_answer(question, docs)
    # Print the answer
    print(answer)