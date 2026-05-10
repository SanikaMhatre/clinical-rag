"""
eval.py — Offline RAGAs evaluation for the Clinical RAG pipeline
Compatible with ragas >= 0.2.0

Judge LLM  : Gemini 2.0 Flash (free, independent from Groq/Llama)
Embeddings : HuggingFace BiomedBERT (free, local)

Setup:
  1. pip install -U ragas langchain-google-genai langchain-huggingface langchain-chroma
  2. Add to .env:  GOOGLE_API_KEY=your_key_here
  3. Run:          python eval.py
"""

import os
import sqlite3
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Validate keys ─────────────────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env")
    print("Get a free key at: https://aistudio.google.com")
    print("Add to .env:  GOOGLE_API_KEY=your_key_here")
    exit(1)

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

DB_PATH = "rag_logs.db"

# ── Gold eval dataset ─────────────────────────────────────────────────────────
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

# ── Load pipeline components ──────────────────────────────────────────────────
print("Loading BiomedBERT embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Loading ChromaDB vector store...")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print("Loading BM25 index...")
with open("bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
bm25      = data["bm25"]
chunks    = data["chunks"]
metadatas = data["metadatas"]

print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Retrieval helpers ─────────────────────────────────────────────────────────
def vector_search(query, k=20):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [doc.page_content for doc, _ in results]

def bm25_search(query, k=20):
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
    merged = reciprocal_rank_fusion(
        vector_search(query, k=20),
        bm25_search(query, k=20)
    )
    if not merged:
        return []
    candidates = merged[:20]
    scores     = reranker.predict([(query, c) for c in candidates])
    ranked     = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]

# ── Load positively-rated logs from DB ───────────────────────────────────────
def load_logged_samples(limit=20):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            SELECT query, answer FROM logs
            WHERE feedback = 1
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,))
        rows = c.fetchall()
        conn.close()
        return [{"question": q, "ground_truth": a} for q, a in rows]
    except Exception:
        return []

# ── Build eval dataset ────────────────────────────────────────────────────────
print("\nBuilding evaluation dataset...")
samples = EVAL_DATASET.copy()

logged = load_logged_samples()
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
    eval_rows.append({
        "question":     q,
        "contexts":     ctxs,
        "answer":       gt,
        "ground_truth": gt
    })

dataset = Dataset.from_list(eval_rows)

# ── Configure Gemini as judge (ragas v0.2+ API) ───────────────────────────────
print("\nConfiguring Gemini 2.0 Flash as judge LLM...")
print("(Gemini judges Llama's output — no self-evaluation bias)\n")

# Gemini LLM for judging
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# ragas v0.2+ uses instantiated metric objects with llm passed directly
faithfulness_metric   = Faithfulness(llm=gemini_llm)
answer_relevancy_metric = AnswerRelevancy(
    llm=gemini_llm,
    embeddings=embeddings   # BiomedBERT for embedding similarity
)
context_recall_metric = ContextRecall(llm=gemini_llm)

# ── Run evaluation ────────────────────────────────────────────────────────────
print("Running RAGAs evaluation...")
print("This calls Gemini for each sample — takes 2-5 minutes.\n")

try:
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            context_recall_metric,
        ],
        run_config=RunConfig(
            max_workers=1,
            timeout=180,
            max_retries=5,
            retry_wait=60   # wait 60s between retries on rate limit
        )
    )
except Exception as e:
    print(f"\nERROR during evaluation: {e}")
    print("\nCommon fixes:")
    print("  - Check GOOGLE_API_KEY is correct in .env")
    print("  - Gemini rate limit: wait 60s and retry")
    print("  - Try: pip install -U ragas langchain-google-genai")
    exit(1)

# ── Print results ─────────────────────────────────────────────────────────────
df = results.to_pandas()

print("\n" + "=" * 60)
print("RAGAs Evaluation Results")
print("Judge LLM  : Gemini 2.0 Flash")
print("Embeddings : BiomedBERT")
print(f"Samples    : {len(df)}")
print("=" * 60)

metric_cols = ["faithfulness", "answer_relevancy", "context_recall"]
available   = [c for c in metric_cols if c in df.columns]

# Per-question table
print("\nPer-question scores:")
for _, row in df.iterrows():
    q = str(row.get("question",""))[:55]
    scores_str = "  ".join(
        f"{m[:5]}: {row[m]:.2f}" for m in available if m in row
    )
    print(f"  {q:<55} {scores_str}")

# Aggregate
print("\n── Aggregate Scores (higher is better, max 1.0) ──")
for metric in available:
    score = df[metric].mean()
    bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"  {metric:<22} {bar}  {score:.3f}")

if len(available) == 3:
    overall = df[available].mean().mean()
    print(f"\n  Overall mean           {overall:.3f}")
    print(f"\n  Interpretation:")
    if overall >= 0.8:
        print("  GOOD  — system is retrieving and answering well")
    elif overall >= 0.6:
        print("  OK    — acceptable but room to improve")
    else:
        print("  POOR  — check chunking, retrieval k, and prompt")

# Save
out_path = "eval_results.csv"
df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")