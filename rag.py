import os
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from groq import Groq

load_dotenv()

# ── 1. Load indexes from disk ─────────────────────────────────────────────────
print("Loading indexes...")

# Load embedding model (same one used in ingest.py)
embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load ChromaDB vector store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Load BM25 index + chunks
with open("bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
bm25       = data["bm25"]
chunks     = data["chunks"]
metadatas  = data["metadatas"]

# Load cross-encoder reranker (downloads ~80MB on first run)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("All indexes loaded. Ready to query.\n")


# ── 2. Hybrid retrieval ───────────────────────────────────────────────────────
def vector_search(query: str, k: int = 20) -> list[dict]:
    """Dense semantic search using ChromaDB."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {"chunk": doc.page_content, "metadata": doc.metadata, "score": score, "source": "vector"}
        for doc, score in results
    ]


def bm25_search(query: str, k: int = 20) -> list[dict]:
    """Sparse keyword search using BM25."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        {"chunk": chunks[i], "metadata": metadatas[i], "score": float(scores[i]), "source": "bm25"}
        for i in top_indices
        if scores[i] > 0  # skip zero-score results
    ]


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF score = sum of 1/(k + rank) across all lists.
    Higher score = better combined rank.
    """
    chunk_scores: dict[str, float] = {}
    chunk_data:   dict[str, dict]  = {}

    for rank, result in enumerate(vector_results):
        chunk = result["chunk"]
        chunk_scores[chunk] = chunk_scores.get(chunk, 0) + 1 / (k + rank + 1)
        chunk_data[chunk]   = result

    for rank, result in enumerate(bm25_results):
        chunk = result["chunk"]
        chunk_scores[chunk] = chunk_scores.get(chunk, 0) + 1 / (k + rank + 1)
        chunk_data[chunk]   = result

    # Sort by RRF score descending
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_data[chunk] for chunk, _ in sorted_chunks]


def hybrid_search(query: str, k: int = 20) -> list[dict]:
    """Run vector + BM25 search and merge with RRF."""
    vector_results = vector_search(query, k=k)
    bm25_results   = bm25_search(query, k=k)
    merged         = reciprocal_rank_fusion(vector_results, bm25_results)
    return merged


# ── 3. Reranking ──────────────────────────────────────────────────────────────
def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank candidates using a cross-encoder.
    Cross-encoder sees (query, chunk) together — much more accurate than
    bi-encoder similarity scores.
    """
    if not candidates:
        return []

    pairs  = [(query, c["chunk"]) for c in candidates]
    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


# ── 4. LLM Generation ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical assistant helping healthcare professionals 
query medical transcription records.

Rules:
- Answer ONLY using the provided context chunks. Do not use outside knowledge.
- If the context does not contain enough information, say "I cannot find enough 
  information in the provided records to answer this question."
- Always cite which specialty the information came from (e.g. "According to a 
  Cardiology note...").
- Be concise and precise. This is a clinical setting.
- Never guess or fabricate medical information."""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate a grounded answer using Groq + Llama 3."""
    if not context_chunks:
        return "No relevant context found for your query."

    # Build context string with source labels
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        specialty = chunk["metadata"].get("specialty", "Unknown")
        context_parts.append(f"[Chunk {i} — {specialty}]\n{chunk['chunk']}")
    context_str = "\n\n".join(context_parts)

    user_message = f"""Context from medical records:
{context_str}

Question: {query}

Answer based only on the context above:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.1,   # low temperature = more factual, less creative
        max_tokens=512
    )
    return response.choices[0].message.content


# ── 5. Full RAG pipeline ──────────────────────────────────────────────────────
def ask(query: str, verbose: bool = False) -> str:
    """
    Full pipeline:
      query → hybrid search → rerank → LLM generation → answer
    """
    print(f"\nQuery: {query}")
    print("-" * 60)

    # Step 1: Hybrid retrieval
    candidates = hybrid_search(query, k=20)
    if verbose:
        print(f"Retrieved {len(candidates)} candidates via hybrid search")

    # Step 2: Rerank
    top_chunks = rerank(query, candidates, top_n=5)
    if verbose:
        print(f"Reranked to top {len(top_chunks)} chunks")
        for i, chunk in enumerate(top_chunks, 1):
            specialty = chunk["metadata"].get("specialty", "?")
            score     = chunk.get("rerank_score", 0)
            print(f"  {i}. [{specialty}] rerank score: {score:.3f}")

    # Step 3: Generate answer
    answer = generate_answer(query, top_chunks)
    print(f"\nAnswer:\n{answer}")
    return answer


# ── 6. Interactive CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Clinical Notes Q&A — powered by hybrid RAG + Groq")
    print("Type 'quit' to exit, 'verbose' to toggle debug info")
    print("=" * 60)

    verbose = False
    while True:
        query = input("\nYour question: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye.")
            break
        if query.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue
        ask(query, verbose=verbose)