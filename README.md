# Clinical RAG Assistant

An AI-powered clinical knowledge assistant that answers medical questions by searching real medical transcription records across 40+ specialties. Built with hybrid retrieval (semantic + keyword search), cross-encoder reranking, and Groq Llama 3 for grounded, source-cited answers.

## Problem Statement

Healthcare professionals often need to quickly query clinical knowledge across specialties — symptoms, diagnoses, medications, procedures — but manual document search is slow and inconsistent. This assistant replaces that process with an AI system grounded in real medical transcriptions, ensuring answers are traceable to actual clinical records.

## Features

- **Hybrid RAG** — BiomedBERT dense retrieval + BM25 keyword search merged via Reciprocal Rank Fusion
- **Cross-encoder reranking** — MiniLM reranker selects the most relevant chunks before generation
- **PII redaction** — Microsoft Presidio anonymizes names, dates, and locations at ingestion
- **Section-aware chunking** — splits clinical notes on section headers (HPI, Assessment, Plan) for coherent chunks
- **Specialty-aware retrieval** — auto-detects medical specialty from query and filters chunks accordingly
- **Conversational memory** — short-term (session) and long-term (cross-session) memory via embedding similarity
- **HITL feedback** — thumbs up/down with correction input, stored to SQLite
- **Chat editing** — edit any past message and re-run the pipeline from that point
- **Conversation history** — saved conversations in sidebar like Claude/ChatGPT
- **Observability** — per-step latency logging, LangSmith tracing, review dashboard
- **Evaluation** — RAGAs metrics (faithfulness, answer relevancy, context recall) with independent judge LLM

## Evaluation Results

Evaluated using RAGAs with Groq Gemma2-9B as independent judge LLM (different model from app's Llama 3.3 70B — eliminates self-evaluation bias):

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 0.667 | Answer stays within retrieved context |
| Answer Relevancy | 0.574 | Answer directly addresses the question |
| Context Recall | 0.725 | Retrieval finds the right chunks |
| **Overall** | **0.655** | |

## Tech Stack

| Component | Tool | Version |
|---|---|---|
| Embedding | BiomedBERT (HuggingFace) | microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext |
| Vector store | ChromaDB | 0.5.20 |
| Keyword search | BM25 | rank-bm25 0.2.2 |
| Reranker | MiniLM cross-encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM (generation) | Groq Llama 3.3 70B | groq 0.37.1 |
| LLM (eval judge) | Groq Gemma2-9B | independent model |
| PII redaction | Microsoft Presidio | 2.2.362 |
| Evaluation | RAGAs | 0.1.21 |
| Observability | LangSmith | 0.1.147 |
| UI | Streamlit | 1.57.0 |
| Orchestration | LangChain | 0.1.20 |

## Dataset

- **MTSamples** — 5,000+ medical transcriptions across 40+ specialties ([Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions))
- **MIMIC-IV-Note** (optional upgrade) — 2M+ real clinical notes, requires PhysioNet credentialing at physionet.org

## Project Structure

```
clinical-rag/
├── ingest.py          # Data loading, PII redaction, chunking, embedding, indexing
├── app.py             # Streamlit UI — chat, HITL, review dashboard, eval dashboard
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
conda create -n rag-eval python=3.11 --no-default-packages -y
conda activate rag-eval
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

**3. Add environment variables — create a `.env` file:**
```
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=clinical-rag
```

**4. Download the dataset:**

Download MTSamples from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) and save as `mtsamples.csv` in the project root.

**5. Run ingestion (once):**
```bash
python ingest.py
```

**6. Run the app:**
```bash
python -m streamlit run app.py
```

**7. Run evaluation (optional):**
```bash
python eval.py
```

## Free API Keys

| Service | Link | Notes |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | Free, no credit card needed |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) | Free tier, 5000 traces/month |

## Important Notes

- **Package versions are pinned** — ragas 0.1.21 requires specific langchain-core and pydantic versions. Do not upgrade these without testing.
- **Run order** — `ingest.py` must complete before running `app.py` or `eval.py`
- **Eval environment** — use the same `rag-eval` conda environment for both app and eval
- **ChromaDB** — the `chroma_db/` folder is gitignored. Each user must run `ingest.py` to generate it locally.