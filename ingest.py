import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import pickle

load_dotenv()

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading MTSamples dataset...")
df = pd.read_csv("mtsamples.csv")
df = df.dropna(subset=["transcription"])
df = df[df["transcription"].str.strip().str.len() > 100].reset_index(drop=True)
print(f"Loaded {len(df)} transcriptions across {df['medical_specialty'].nunique()} specialties")

# ── 2. PII Redaction ──────────────────────────────────────────────────────────
print("Setting up PII redaction...")
analyzer   = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> str:
    try:
        results    = analyzer.analyze(
            text=text,
            entities=["PERSON", "DATE_TIME", "LOCATION", "PHONE_NUMBER", "EMAIL_ADDRESS"],
            language="en"
        )
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except Exception:
        return text

print("Redacting PII...")
df["transcription_clean"] = df["transcription"].apply(redact_pii)
print("PII redaction complete.")

# ── 3. Section-aware chunking ─────────────────────────────────────────────────
# Clinical notes follow predictable section headers.
# We split on these first, keeping each section as a unit,
# then further split only if a section is too long.

CLINICAL_HEADERS = [
    r"CHIEF COMPLAINT",
    r"HISTORY OF PRESENT ILLNESS",
    r"HPI",
    r"PAST MEDICAL HISTORY",
    r"PAST SURGICAL HISTORY",
    r"MEDICATIONS",
    r"ALLERGIES",
    r"REVIEW OF SYSTEMS",
    r"PHYSICAL EXAMINATION",
    r"ASSESSMENT",
    r"PLAN",
    r"IMPRESSION",
    r"DIAGNOSIS",
    r"PROCEDURES?",
    r"FINDINGS",
    r"DISCHARGE",
    r"LABORATORY",
    r"RADIOLOGY",
    r"RECOMMENDATIONS?",
    r"FOLLOW.?UP",
]

HEADER_PATTERN = re.compile(
    r"(?im)^(" + "|".join(CLINICAL_HEADERS) + r")[:\s]*$"
)

def section_aware_split(text: str, max_chunk: int = 600) -> list[str]:
    """
    Split clinical text on section headers first.
    If a section is still too long, fall back to RecursiveCharacterTextSplitter.
    """
    # Split on headers — keep the header with its content
    parts   = HEADER_PATTERN.split(text)
    sections = []

    # parts alternates: [pre-header text, header, content, header, content, ...]
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if chunk:
            sections.append(chunk)
        i += 1

    # Fallback splitter for long sections
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )

    final_chunks = []
    for section in sections:
        if len(section) <= max_chunk:
            if len(section.strip()) > 50:
                final_chunks.append(section.strip())
        else:
            sub_chunks = splitter.split_text(section)
            final_chunks.extend([c for c in sub_chunks if len(c.strip()) > 50])

    return final_chunks

print("Chunking transcriptions with section-aware splitting...")
chunks    = []
metadatas = []

for _, row in df.iterrows():
    text      = row["transcription_clean"]
    specialty = str(row.get("medical_specialty", "unknown")).strip()
    desc      = str(row.get("description", "")).strip()

    doc_chunks = section_aware_split(text)
    for chunk in doc_chunks:
        chunks.append(chunk)
        metadatas.append({
            "specialty": specialty,
            "description": desc,
        })

print(f"Created {len(chunks)} chunks from {len(df)} documents")
print(f"Average chunk length: {sum(len(c) for c in chunks) // len(chunks)} chars")

# ── 4. Embedding + Vector Store ───────────────────────────────────────────────
print("Loading BiomedBERT embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding chunks and storing in ChromaDB...")
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("ChromaDB saved to ./chroma_db")

# ── 5. BM25 Index ─────────────────────────────────────────────────────────────
print("Building BM25 index...")
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

with open("bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunks": chunks, "metadatas": metadatas}, f)
print("BM25 index saved.")

print("\nIngestion complete! Run: streamlit run app.py")