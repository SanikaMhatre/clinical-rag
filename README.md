# Clinical RAG Assistant

An AI-powered clinical knowledge assistant that answers medical questions by searching real medical transcription records across 40+ specialties. Built with hybrid retrieval (semantic + keyword search), cross-encoder reranking, and Groq Llama 3 for grounded, source-cited answers.

## Problem Statement

Healthcare professionals often need to quickly query clinical knowledge across specialties — symptoms, diagnoses, medications, procedures — but manual document search is slow and inconsistent. This assistant replaces that process with an AI system grounded in real medical transcriptions, ensuring answers are traceable to actual clinical records.

## Features

- **Hybrid RAG** — BiomedBERT dense retrieval + BM25 keyword search merged via Reciprocal Rank Fusion
- **Cross-encoder reranking** — MiniLM reranker selects the most relevant chunks before generation
- **PII redaction** — Microsoft Presidio anonymizes names, dates, and locations at ingestion
- **Specialty-aware retrieval** — auto-detects medical specialty from query and filters chunks accordingly
- **Conversational memory** — short-term (session) and long-term (cross-session) memory
- **HITL feedback** — thumbs up/down with correction input, stored to SQLite
- **Observability** — per-step latency logging, LangSmith tracing, review dashboard
- **Evaluation** — RAGAs metrics (faithfulness, answer relevancy, context recall) with Gemini as independent judge

## Tech Stack

| Component | Tool |
|---|---|
| Embedding | BiomedBERT (HuggingFace) |
| Vector store | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq Llama 3.3 70B |
| PII redaction | Microsoft Presidio |
| Evaluation | RAGAs + Gemini 1.5 Flash |
| Observability | LangSmith |
| UI | Streamlit |

## Dataset

- **MTSamples** — 5,000+ medical transcriptions across 40+ specialties ([Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions))
- **MIMIC-IV-Note** (optional upgrade) — 2M+ real clinical notes, requires PhysioNet access

## Project Structure

```
clinical-rag/
├── ingest.py          # Data loading, PII redaction, chunking, embedding, indexing
├── app.py             # Streamlit UI — chat, HITL feedback, review dashboard
├── rag.py             # CLI query pipeline
├── eval.py            # RAGAs evaluation script
├── .streamlit/
│   └── config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repo:**
```bash
git clone https://github.com/SanikaMhatre/clinical-rag.git
cd clinical-rag
```

**2. Create conda environment:**
```bash
conda create -n clinical-rag python=3.11
conda activate clinical-rag
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

**3. Add environment variables — create a `.env` file:**
```
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=clinical-rag
GOOGLE_API_KEY=your_gemini_key_here
```

**4. Download the dataset:**

Download MTSamples from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) and save as `mtsamples.csv` in the project root.

**5. Run ingestion (once):**
```bash
python ingest.py
```

**6. Run the app:**
```bash
streamlit run app.py
```

**7. Run evaluation (optional):**
```bash
python eval.py
```

## Free API Keys

| Service | Link | Notes |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | Free, no credit card |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) | Free tier, 5000 traces/month |
| Gemini | [aistudio.google.com](https://aistudio.google.com) | Free tier, for eval only |

## Evaluation

RAGAs metrics scored with Gemini 1.5 Flash as independent judge (eliminates self-evaluation bias from using the same LLM for generation and judging):

| Metric | Description |
|---|---|
| Faithfulness | Does the answer stick to the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Recall | Did retrieval find the right chunks? |

Results saved to `eval_results.csv` after each run.
