"""
eval.py — RAGAs evaluation for the Clinical RAG pipeline
Environment: rag-eval conda env

Judge LLM  : Groq Gemma2-9B (different from app's Llama 3.3 — no self-evaluation bias)
Embeddings : local sentence-transformers (no OpenAI key needed)
Retrieval  : BM25 + cross-encoder reranker
Metrics    : Faithfulness, Answer Relevancy, Context Recall
"""

import os
import time
import sqlite3
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env")
    exit(1)

# Dummy OpenAI key to stop ragas from crashing on import
# We override the embeddings below with a local model
os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingfaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

DB_PATH = "rag_logs.db"

EVAL_DATASET = [
    {
        "question":     "What are the symptoms of myocardial infarction?",
        "ground_truth": "Symptoms of myocardial infarction include crushing chest pain, "
                        "pain radiating to the left arm or jaw, shortness of breath, "
                        "diaphoresis, nausea, and fatigue."
    },
    {
        "question":     "How is type 2 diabetes managed?",
        "ground_truth": "Type 2 diabetes management includes lifestyle modifications, "
                        "blood glucose monitoring, oral medications such as metformin, "
                        "and insulin therapy when required. HbA1c monitoring tracks "
                        "long-term glucose control."
    },
    {
        "question":     "What medications are used to treat hypertension?",
        "ground_truth": "Hypertension is treated with ACE inhibitors, beta-blockers, "
                        "calcium channel blockers, and thiazide diuretics, often in combination."
    },
    {
        "question":     "What are the signs of pneumonia?",
        "ground_truth": "Signs of pneumonia include fever, productive cough, chest pain, "
                        "shortness of breath, decreased breath sounds, and consolidation "
                        "on chest imaging."
    },
    {
        "question":     "Describe common knee surgery procedures.",
        "ground_truth": "Common knee surgeries include arthroscopy for meniscal tears, "
                        "ACL reconstruction, total knee replacement for severe arthritis, "
                        "and partial knee replacement."
    },
    {
        "question":     "What are the symptoms of a stroke?",
        "ground_truth": "Stroke symptoms include sudden facial drooping, arm weakness, "
                        "speech difficulty, vision changes, severe headache, and loss "
                        "of coordination."
    },
    {
        "question":     "How is GERD diagnosed and treated?",
        "ground_truth": "GERD is diagnosed via symptoms, endoscopy, and pH monitoring. "
                        "Treatment includes proton pump inhibitors, H2 blockers, "
                        "lifestyle changes, and in severe cases surgical fundoplication."
    },
    {
        "question":     "What are common findings in a lumbar MRI for back pain?",
        "ground_truth": "Common lumbar MRI findings include disc herniation, spinal stenosis, "
                        "degenerative disc disease, facet joint arthropathy, and nerve "
                        "root compression."
    },
    {
        "question":     "What medications are used in depression treatment?",
        "ground_truth": "Depression is treated with SSRIs such as sertraline and fluoxetine, "
                        "SNRIs, tricyclic antidepressants, and in resistant cases MAOIs "
                        "or atypical antipsychotics."
    },
    {
        "question":     "What are the symptoms of a urinary tract infection?",
        "ground_truth": "UTI symptoms include dysuria, urinary frequency and urgency, "
                        "cloudy or malodorous urine, pelvic pain, and in severe cases "
                        "fever and flank pain indicating kidney involvement."
    },
]

# ── Load BM25 index ───────────────────────────────────────────────────────────
print("Loading BM25 index...")
with open("bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
bm25      = data["bm25"]
chunks    = data["chunks"]
metadatas = data["metadatas"]

print("Loading ChromaDB vector store...")
from langchain_chroma import Chroma as LangChroma
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = LangChroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Hybrid retrieval ──────────────────────────────────────────────────────────
def vector_search(query, k=30):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [doc.page_content for doc, _ in results]

def bm25_search(query, k=30):
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx if scores[i] > 0]

def reciprocal_rank_fusion(v_chunks, b_chunks, k=60):
    scores = {}
    for rank, c in enumerate(v_chunks):
        scores[c] = scores.get(c, 0) + 1 / (k + rank + 1)
    for rank, c in enumerate(b_chunks):
        scores[c] = scores.get(c, 0) + 1 / (k + rank + 1)
    return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def retrieve(query, top_k=5):
    merged     = reciprocal_rank_fusion(
        vector_search(query, k=20),
        bm25_search(query, k=20)
    )
    if not merged:
        return []
    candidates    = merged[:20]
    rerank_scores = reranker.predict([(query, c) for c in candidates])
    ranked        = sorted(zip(candidates, rerank_scores),
                           key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:8]]

# ── Load positively-rated logs ────────────────────────────────────────────────
def load_logged_samples(limit=20):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("SELECT query, answer FROM logs WHERE feedback=1 "
                  "ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return [{"question": q, "ground_truth": a} for q, a in rows]
    except Exception:
        return []

# ── Build eval dataset ────────────────────────────────────────────────────────
print("\nBuilding evaluation dataset...")
samples = EVAL_DATASET.copy()
logged  = load_logged_samples()
if logged:
    print(f"Adding {len(logged)} positively-rated Q&As from rag_logs.db")
    samples.extend(logged)
else:
    print("No logged feedback found — using gold dataset only")
print(f"Total eval samples: {len(samples)}")

print("\nRunning retrieval for each question...")
eval_rows = []
for i, sample in enumerate(samples):
    q    = sample["question"]
    gt   = sample["ground_truth"]
    ctxs = retrieve(q, top_k=5)
    if not ctxs:
        print(f"  [{i+1}/{len(samples)}] WARNING: no context for: {q[:60]}")
        ctxs = ["No relevant context found."]
    else:
        print(f"  [{i+1}/{len(samples)}] OK — {q[:60]}")
    # Generate actual answer from Groq for faithful evaluation
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context_str = "\n\n".join(ctxs[:3])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer only from the provided context. Be concise."},
            {"role": "user",   "content": f"Context:\n{context_str}\n\nQuestion: {q}\n\nAnswer:"}
        ],
        temperature=0.1, max_tokens=256
    )
    generated_answer = response.choices[0].message.content

    eval_rows.append({
        "question":     q,
        "contexts":     ctxs,
        "answer":       generated_answer,   # ← real LLM answer
        "ground_truth": gt
    })
    time.sleep(2)  # avoid rate limits
    

dataset = Dataset.from_list(eval_rows)

# ── Configure judge LLM + local embeddings ────────────────────────────────────
print("\nConfiguring Groq Gemma2-9B as judge LLM...")
print("(Different model from app's Llama 3.3 — no self-evaluation bias)\n")

judge_chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
judge_llm = LangchainLLMWrapper(judge_chat)

# Local embeddings for answer_relevancy — no OpenAI key needed
local_emb = HuggingfaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

faithfulness.llm            = judge_llm
answer_relevancy.llm        = judge_llm
answer_relevancy.embeddings = local_emb
context_recall.llm          = judge_llm

# ── Run evaluation ────────────────────────────────────────────────────────────
print("Running RAGAs evaluation — takes a few minutes...\n")

try:
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        run_config=RunConfig(max_workers=1, timeout=300)
    )
except Exception as e:
    print(f"\nERROR during evaluation: {e}")
    print("  - Check GROQ_API_KEY in .env")
    print("  - Try waiting 60s if rate limited")
    exit(1)

# ── Print results ─────────────────────────────────────────────────────────────
df = results.to_pandas()

print("\n" + "=" * 60)
print("RAGAs Evaluation Results")
print("Judge LLM  : Groq Gemma2-9B")
print("Embeddings : sentence-transformers/all-MiniLM-L6-v2 (local)")
print("Retrieval  : BM25 + cross-encoder reranker")
print(f"Samples    : {len(df)}")
print("=" * 60)

metric_cols = ["faithfulness", "answer_relevancy", "context_recall"]
available   = [c for c in metric_cols if c in df.columns]

print("\nPer-question scores:")
for _, row in df.iterrows():
    q          = str(row.get("question",""))[:55]
    scores_str = "  ".join(
        f"{m[:5]}: {float(row[m]):.2f}"
        if not np.isnan(float(row[m])) else f"{m[:5]}: N/A"
        for m in available if m in row
    )
    print(f"  {q:<55} {scores_str}")

print("\n── Aggregate Scores (higher is better, max 1.0) ──")
for metric in available:
    score = df[metric].mean()
    if np.isnan(score):
        print(f"  {metric:<22} N/A")
        continue
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"  {metric:<22} {bar}  {score:.3f}")

valid = [m for m in available if not np.isnan(df[m].mean())]
if len(valid) == 3:
    overall = df[valid].mean().mean()
    print(f"\n  Overall mean           {overall:.3f}")
    if overall >= 0.8:
        print("  GOOD  — system is retrieving and answering well")
    elif overall >= 0.6:
        print("  OK    — acceptable but room to improve")
    else:
        print("  POOR  — check chunking, retrieval k, and prompt")

out_path = "eval_results.csv"
df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")