"""
Simple RAG pipeline with Groq API: Query a PDF with Retrieval + LLM Answering
"""

import os
from dotenv import load_dotenv
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load LLM client (Groq)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into manageable chunks for embeddings."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """Create FAISS index for document chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, chunks

def retrieve(query: str, index, chunks, model_name: str = "all-MiniLM-L6-v2", top_k: int = 3) -> list[str]:
    """Fetch top-k relevant chunks for a query."""
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query])
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]

def query_llm(query: str, context_chunks: list[str]) -> str:
    """Query the Groq LLM with retrieved context."""
    context = "\n".join(context_chunks)
    prompt = f"You are a helpful assistant. Use the following context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Groq LLM query failed: {e}")

if __name__ == "__main__":
    print("\n================ RAG Pipeline Execution Started ================\n")

    # Step 1: Extract text from PDF
    print("Step 1: Extracting text from the PDF document...")
    pdf_text = extract_text_from_pdf("sample.pdf")
    print("Text extraction complete. Proceeding to chunk the text for embedding.\n")

    # Step 2: Chunk text for embeddings
    chunks = chunk_text(pdf_text)
    print(f"Step 2: Text has been split into {len(chunks)} chunks.\n")

    # Step 3: Build FAISS index
    print("Step 3: Building FAISS index for efficient similarity search.")
    index, embeddings, chunks = build_faiss_index(chunks)
    print("FAISS index construction complete.\n")

    # Step 4: Retrieve relevant chunks
    query = "What are the key takeaways from this document?"
    print(f"Step 4: Retrieving relevant document chunks for the query: '{query}'")
    relevant_chunks = retrieve(query, index, chunks)
    print(f"Retrieved {len(relevant_chunks)} relevant chunks. Passing context to the LLM.\n")

    # Step 5: Query LLM
    answer = query_llm(query, relevant_chunks)
    print("Step 5: LLM response received. Displaying the answer below:\n")
    print("========================= Answer =========================")
    print("Answer:", answer)
    print("==========================================================\n")
