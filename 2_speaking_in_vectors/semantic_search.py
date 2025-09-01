"""
Semantic Search with Embeddings and FAISS
-----------------------------------------
Steps:
1. Generate embeddings using Sentence-Transformers
2. Build a FAISS index for similarity search
3. Perform queries with cosine similarity
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Define sample documents
documents = [
    "The jaguar is a large cat native to the Americas.",
    "Jaguar is a luxury car brand from the UK.",
    "The Jacksonville Jaguars are an NFL team.",
    "Tigers are also big cats found in Asia."
]

# Step 3: Generate embeddings
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Step 4: Create FAISS index
dimension = doc_embeddings.shape[1]  # Embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(doc_embeddings)

# Step 5: Perform search
def semantic_search(query: str, top_k: int = 2):
    """
    Search documents by semantic similarity.

    Args:
        query (str): The user's search query
        top_k (int): Number of top results to return

    Returns:
        List of (document, score) pairs
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [(documents[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    return results

# Example search
if __name__ == "__main__":
    query = "Tell me about big cats"
    results = semantic_search(query)
    for doc, score in results:
        print(f"Doc: {doc} | Distance: {score:.4f}")