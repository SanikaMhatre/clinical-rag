import os
import time
import sqlite3
import pickle
import json
import html
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from groq import Groq
import logging

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2",  os.getenv("LANGCHAIN_TRACING_V2",  "false"))
os.environ.setdefault("LANGCHAIN_API_KEY",      os.getenv("LANGCHAIN_API_KEY",     ""))
os.environ.setdefault("LANGCHAIN_PROJECT",      os.getenv("LANGCHAIN_PROJECT",     "clinical-rag"))

logging.basicConfig(
    filename="app.log", level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Clinical Notes Q&A", layout="wide")
st.markdown("""
<style>
  .source-card{background:var(--color-background-secondary);border-left:3px solid #378ADD;border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:13px}
  .specialty-badge{display:inline-block;background:#E6F1FB;color:#0C447C;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:500;margin-bottom:6px}
  .score-text{color:#888;font-size:11px;float:right}
  .warning-box{background:#FAEEDA;border-left:3px solid #BA7517;border-radius:6px;padding:10px 14px;font-size:13px;color:#633806;margin-bottom:8px}
  .latency-text{font-size:11px;color:#888;margin-top:4px}
  .hitl-box{background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:8px;padding:14px;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SHORT_TERM_TURNS     = 5
LONG_TERM_RESULTS    = 3
CONFIDENCE_THRESHOLD = -2.0
DB_PATH              = "rag_logs.db"
CONV_DB_PATH         = "conversations.db"

ALL_SPECIALTIES = [
    "Allergy / Immunology","Autopsy","Bariatrics","Cardiovascular / Pulmonary",
    "Chiropractic","Consult - History and Phy.","Cosmetic / Plastic Surgery",
    "Dentistry","Dermatology","Diets and Nutritions","Discharge Summary",
    "Emergency Room Reports","Endocrinology","ENT - Otolaryngology",
    "Gastroenterology","General Medicine","Hematology - Oncology",
    "Hospice - Palliative Care","IME-QME-Work Comp etc.","Lab Medicine - Pathology",
    "Letters","Nephrology","Neurology","Neurosurgery","Obstetrics / Gynecology",
    "Office Notes","Ophthalmology","Orthopedic","Pain Management",
    "Pediatrics - Neonatal","Physical Medicine - Rehab","Podiatry",
    "Psychiatry / Psychology","Pulmonology","Radiology","Rheumatology",
    "Sleep Medicine","SOAP / Chart / Progress Notes","Speech - Language",
    "Surgery","Urology"
]

SPECIALTY_MAP = {
    "Cardiovascular / Pulmonary": [
        "heart","cardiac","myocardial","infarction","mi","angina","coronary",
        "chest pain","ecg","ekg","arrhythmia","hypertension","blood pressure",
        "lung","pulmonary","respiratory","breathing","shortness of breath",
        "copd","asthma","pneumonia"
    ],
    "Endocrinology": [
        "diabetes","insulin","glucose","thyroid","hormone","hba1c",
        "blood sugar","endocrine","metformin","lantus","hypothyroid"
    ],
    "Neurology": [
        "brain","stroke","seizure","headache","migraine","nerve","neurological",
        "alzheimer","parkinson","dementia","epilepsy","tremor","neuropathy","spinal"
    ],
    "Orthopedic": [
        "bone","fracture","joint","knee","hip","shoulder","spine","arthritis",
        "orthopedic","ligament","tendon","cartilage","back pain","lumbar","cervical"
    ],
    "Gastroenterology": [
        "stomach","bowel","colon","liver","hepatitis","gastro","nausea",
        "vomiting","diarrhea","constipation","abdomen","ulcer","reflux","gerd","ibs"
    ],
    "Psychiatry / Psychology": [
        "depression","anxiety","mental","psychiatric","psychology","bipolar",
        "schizophrenia","ptsd","adhd","sleep disorder","insomnia","mood","psychosis"
    ],
    "Urology": [
        "kidney","bladder","urine","urinary","prostate","renal","nephrology",
        "uti","incontinence","dialysis","catheter"
    ],
    "Obstetrics / Gynecology": [
        "pregnancy","obstetric","gynecology","uterus","ovary","menstrual",
        "cervix","prenatal","labor","delivery","fetal"
    ],
    "Radiology": [
        "xray","x-ray","mri","ct scan","imaging","ultrasound","radiology",
        "scan","contrast","radiograph"
    ],
    "Surgery": [
        "surgery","surgical","incision","laparoscopic","resection","anesthesia",
        "postoperative","preoperative","excision","biopsy"
    ],
    "Hematology - Oncology": [
        "cancer","tumor","chemotherapy","radiation","oncology","leukemia",
        "lymphoma","anemia","blood disorder","malignant"
    ],
    "ENT - Otolaryngology": [
        "ear","nose","throat","ent","sinus","tonsil","hearing","nasal",
        "otolaryngology","larynx","pharynx","tinnitus","vertigo"
    ],
    "Sleep Medicine": [
        "sleep","snoring","apnea","insomnia","somnolence","fatigue",
        "restless","narcolepsy","cpap","polysomnography"
    ],
}

FOLLOWUP_SIGNALS = [
    "it","this","that","these","those","the condition","the disease",
    "the medication","the treatment","the surgery","the procedure",
    "tell me more","explain more","what about","how about","and","also",
    "what else","go on","continue","elaborate","further","additionally"
]

def detect_specialty(query: str) -> list[str]:
    q = query.lower()
    return [sp for sp, kws in SPECIALTY_MAP.items() if any(kw in q for kw in kws)]

def is_followup(query: str, has_history: bool) -> bool:
    if not has_history:
        return False
    q_lower = query.lower().strip()
    if len(q_lower.split()) <= 4:
        return True
    return any(sig in q_lower for sig in FOLLOWUP_SIGNALS)

# ── LangSmith manual logging ──────────────────────────────────────────────────
def log_to_langsmith(query, answer, top_chunks, specialty_filter,
                     t0, t1, t2, t3, short_term, long_term):
    try:
        if os.getenv("LANGCHAIN_TRACING_V2","false").lower() != "true":
            return
        if not os.getenv("LANGCHAIN_API_KEY",""):
            return
        from langsmith import Client as LangSmithClient
        ls = LangSmithClient()
        ls.create_run(
            name         = f"clinical-rag | {query[:50]}",
            run_type     = "chain",
            project_name = os.getenv("LANGCHAIN_PROJECT","clinical-rag"),
            start_time   = datetime.fromtimestamp(t0, tz=timezone.utc),
            inputs       = {"query": query, "specialty_filter": specialty_filter},
            outputs      = {
                "answer":           answer,
                "chunks_used":      [c["chunk"][:200] for c in top_chunks],
                "specialties":      [c["metadata"].get("specialty","?") for c in top_chunks],
                "rerank_scores":    [round(c.get("rerank_score",0),3) for c in top_chunks],
                "top_rerank_score": round(top_chunks[0]["rerank_score"] if top_chunks else -99,3),
                "short_term_turns": len(short_term),
                "long_term_hits":   len(long_term),
            },
            extra        = {
                "latency": {
                    "retrieval_ms": round((t1-t0)*1000),
                    "rerank_ms":    round((t2-t1)*1000),
                    "llm_ms":       round((t3-t2)*1000),
                    "total_ms":     round((t3-t0)*1000),
                }
            },
            end_time     = datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.warning(f"LangSmith logging failed: {e}")

# ── Conversations DB ──────────────────────────────────────────────────────────
def init_conv_db():
    conn = sqlite3.connect(CONV_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT,
            created_at TEXT, updated_at TEXT, messages TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_conversation(conv_id, title, messages):
    clean = []
    for m in messages:
        entry = {"role": m["role"], "content": m["content"]}
        for k in ("low_conf","latency","st_used","lt_used"):
            if k in m:
                entry[k] = m[k]
        if "sources" in m:
            entry["sources"] = [
                {"chunk": c["chunk"], "metadata": c["metadata"],
                 "rerank_score": float(c.get("rerank_score",0)),
                 "source": c.get("source","")}
                for c in m["sources"]
            ]
        clean.append(entry)
    conn = sqlite3.connect(CONV_DB_PATH)
    now  = datetime.now().isoformat()
    exist = conn.execute("SELECT id FROM conversations WHERE id=?", (conv_id,)).fetchone()
    if exist:
        conn.execute("UPDATE conversations SET title=?, updated_at=?, messages=? WHERE id=?",
                     (title, now, json.dumps(clean), conv_id))
    else:
        conn.execute("INSERT INTO conversations VALUES (?,?,?,?,?)",
                     (conv_id, title, now, now, json.dumps(clean)))
    conn.commit()
    conn.close()

def get_all_conversations():
    conn = sqlite3.connect(CONV_DB_PATH)
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return rows

def load_conversation(conv_id):
    conn = sqlite3.connect(CONV_DB_PATH)
    row  = conn.execute("SELECT messages FROM conversations WHERE id=?", (conv_id,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else []

def delete_conversation(conv_id):
    conn = sqlite3.connect(CONV_DB_PATH)
    conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()

# ── RAG logs DB ───────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, query TEXT, answer TEXT,
            chunks_used TEXT, specialties TEXT,
            retrieval_ms REAL, rerank_ms REAL, llm_ms REAL, total_ms REAL,
            top_rerank_score REAL, specialty_filter TEXT,
            feedback INTEGER DEFAULT NULL, correction TEXT DEFAULT NULL,
            query_embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def log_response(query, answer, chunks, r_ms, rk_ms, l_ms, tot_ms,
                 score, sf, query_embedding=None):
    specialties = list({c["metadata"].get("specialty","Unknown") for c in chunks})
    emb_bytes   = query_embedding.tobytes() if query_embedding is not None else None
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO logs (timestamp,query,answer,chunks_used,specialties,
                          retrieval_ms,rerank_ms,llm_ms,total_ms,top_rerank_score,
                          specialty_filter,query_embedding)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (datetime.now().isoformat(), query, answer,
          " | ".join(c["chunk"][:200] for c in chunks),
          ", ".join(specialties), r_ms, rk_ms, l_ms, tot_ms, score,
          ", ".join(sf) if sf else "none", emb_bytes))
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return row_id

def save_feedback(row_id, feedback, correction=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE logs SET feedback=?, correction=? WHERE id=?",
                 (feedback, correction, row_id))
    conn.commit()
    conn.close()

def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY timestamp DESC")
    rows = c.fetchall()
    cols = [d[0] for d in c.description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]

def get_recent_logs(n=SHORT_TERM_TURNS):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT query, answer FROM logs ORDER BY timestamp DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

def get_similar_past_qa(query_embedding, top_n=LONG_TERM_RESULTS,
                        exclude_last_n=SHORT_TERM_TURNS):
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT query, answer, correction, query_embedding FROM logs
        WHERE query_embedding IS NOT NULL
        ORDER BY timestamp DESC LIMIT -1 OFFSET ?
    """, (exclude_last_n,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    scored = []
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    for q, a, corr, emb_bytes in rows:
        try:
            past_emb  = np.frombuffer(emb_bytes, dtype=np.float32)
            past_norm = past_emb / (np.linalg.norm(past_emb) + 1e-9)
            sim       = float(np.dot(q_norm, past_norm))
            scored.append({"query": q, "answer": a, "correction": corr, "similarity": sim})
        except Exception:
            continue
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return [r for r in scored[:top_n] if r["similarity"] > 0.7]

init_db()
init_conv_db()

# ── Guardrail ─────────────────────────────────────────────────────────────────
MEDICAL_KEYWORDS = [
    "patient","symptom","diagnosis","treatment","medication","surgery","disease",
    "condition","procedure","doctor","clinical","medical","hospital","therapy",
    "drug","dose","pain","infection","blood","heart","lung","brain","kidney",
    "liver","cancer","diabetes","hypertension","test","lab","result","history",
    "exam","physical","report","note","discharge","health","nurse","physician",
    "chest","cardiac","respiratory","orthopedic","neurology","gastro","urology",
    "sleep","ear","nose","throat","sinus","bone","joint","fracture"
]

def is_medical_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in MEDICAL_KEYWORDS) or len(q.split()) >= 3

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_indexes():
    embeddings  = HuggingFaceEmbeddings(
        model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    with open("bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return vectorstore, data["bm25"], data["chunks"], data["metadatas"], reranker, groq_client, embeddings

# ── RAG functions ─────────────────────────────────────────────────────────────
def vector_search(vectorstore, query, k=30, specialty_filter=None):
    kwargs = {"k": k}
    if specialty_filter:
        kwargs["filter"] = {"specialty": {"$in": specialty_filter}}
    results = vectorstore.similarity_search_with_score(query, **kwargs)
    return [{"chunk": doc.page_content, "metadata": doc.metadata,
             "score": score, "source": "vector"} for doc, score in results]

def bm25_search(bm25, chunks, metadatas, query, k=30, specialty_filter=None):
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1]
    results = []
    for i in top_idx:
        if scores[i] <= 0:
            break
        if specialty_filter and metadatas[i].get("specialty","") not in specialty_filter:
            continue
        results.append({"chunk": chunks[i], "metadata": metadatas[i],
                        "score": float(scores[i]), "source": "bm25"})
        if len(results) >= k:
            break
    return results

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    chunk_scores, chunk_data = {}, {}
    for rank, r in enumerate(vector_results):
        c = r["chunk"]
        chunk_scores[c] = chunk_scores.get(c, 0) + 1 / (k + rank + 1)
        chunk_data[c]   = r
    for rank, r in enumerate(bm25_results):
        c = r["chunk"]
        chunk_scores[c] = chunk_scores.get(c, 0) + 1 / (k + rank + 1)
        chunk_data[c]   = r
    return [chunk_data[c] for c, _ in sorted(chunk_scores.items(),
                                              key=lambda x: x[1], reverse=True)]

def rerank_chunks(reranker, query, candidates, top_n=5):
    if not candidates:
        return []
    scores = reranker.predict([(query, c["chunk"]) for c in candidates])
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

SYSTEM_PROMPT = """You are a clinical knowledge assistant helping healthcare professionals.
You are given excerpts from real medical transcription records as context.

STRICT RULES:
- Answer ONLY using information explicitly stated in the provided context.
- Do NOT synthesize, infer, assume, or use any outside knowledge whatsoever.
- Do NOT say things like "it is implied" or "it can be inferred" — only state what
  is directly written in the context.
- If the context contains partial information, use what is available and note it
  may be incomplete. Only respond with "I cannot find sufficient information"
  if the context is completely unrelated to the question.
- Do NOT describe specific patients or individuals. Do not say "the patient", "a patient",
  "she", "he", or refer to personal cases.
- Transform any patient-specific text into general clinical facts only if the fact
  is explicitly stated — not inferred.
  Example: "patient presented with crushing chest pain" →
  "Crushing chest pain is documented as a presentation."
- Write in clean natural prose. Do not mention chunk numbers.
- Use conversation history only to resolve pronouns like "it" or "that condition"
  — do not use it to add information not in the current context.
- Be concise and precise. Never fabricate information."""

def build_messages(query, context_chunks, short_term, long_term, session_history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if long_term:
        lt_text = "\n\n".join(
            f"Past Q: {m['query']}\nPast A: {m['correction'] or m['answer']}"
            for m in long_term
        )
        messages.append({"role": "system", "content": f"LONG TERM MEMORY:\n{lt_text}"})
    history_turns = session_history or short_term
    for item in history_turns[-SHORT_TERM_TURNS:]:
        if isinstance(item, dict):
            if item["role"] in ("user","assistant"):
                messages.append({"role": item["role"], "content": item["content"]})
        else:
            messages.append({"role": "user",      "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})
    context_str = "\n\n".join(
        f"[{c['metadata'].get('specialty','Unknown')}]\n{c['chunk']}"
        for c in context_chunks
    )
    messages.append({"role": "user", "content":
        f"Clinical context (facts only, do not describe patients):\n"
        f"{context_str}\n\nQuestion: {query}\n\nAnswer:"})
    return messages

def generate_answer(groq_client, query, context_chunks,
                    short_term, long_term, session_history=None):
    if not context_chunks:
        return "No relevant context found."
    messages = build_messages(query, context_chunks, short_term, long_term, session_history)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content

def run_rag(query, vectorstore, bm25, chunks, metadatas, reranker,
            groq_client, embeddings_model, top_k,
            specialty_filter=None, session_history=None):

    query_embedding = np.array(
        embeddings_model.embed_query(query), dtype=np.float32)

    t0     = time.time()
    merged = reciprocal_rank_fusion(
        vector_search(vectorstore, query, k=30, specialty_filter=specialty_filter),
        bm25_search(bm25, chunks, metadatas, query, k=30, specialty_filter=specialty_filter)
    )
    t1         = time.time()
    top_chunks = rerank_chunks(reranker, query, merged, top_n=top_k)
    t2         = time.time()
    short_term = get_recent_logs(n=SHORT_TERM_TURNS)
    long_term  = get_similar_past_qa(query_embedding, top_n=LONG_TERM_RESULTS)
    answer     = generate_answer(groq_client, query, top_chunks,
                                 short_term, long_term, session_history)
    t3         = time.time()

    log_to_langsmith(query, answer, top_chunks, specialty_filter,
                     t0, t1, t2, t3, short_term, long_term)

    top_score = top_chunks[0]["rerank_score"] if top_chunks else -99.0
    return (answer, top_chunks, query_embedding,
            (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t3-t0)*1000,
            top_score, top_score < CONFIDENCE_THRESHOLD,
            len(short_term), len(long_term))

# ── Load indexes ──────────────────────────────────────────────────────────────
with st.spinner("Loading models and indexes..."):
    try:
        vectorstore, bm25, chunks, metadatas, reranker_model, groq_client, emb_model = load_indexes()
    except Exception as e:
        st.error(f"Failed to load indexes: {e}")
        st.info("Make sure ingest.py has been run first.")
        st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
def new_conv_id():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

for key, default in [
    ("conv_id",          new_conv_id()),
    ("messages",         []),
    ("log_ids",          []),
    ("pending_query",    None),
    ("specialty_filter", None),
    ("hitl_state",       None),
    ("active_filter",    None),
    ("run_query",        None),
    ("editing_index",    None),
    ("editing_text",     None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def get_conv_title(messages):
    for m in messages:
        if m["role"] == "user":
            t = m["content"][:40]
            return t + ("..." if len(m["content"]) > 40 else "")
    return "New conversation"

def switch_conversation(conv_id):
    if st.session_state.messages:
        save_conversation(st.session_state.conv_id,
                         get_conv_title(st.session_state.messages),
                         st.session_state.messages)
    st.session_state.conv_id          = conv_id
    st.session_state.messages         = load_conversation(conv_id)
    st.session_state.log_ids          = []
    st.session_state.pending_query    = None
    st.session_state.hitl_state       = None
    st.session_state.active_filter    = None
    st.session_state.run_query        = None
    st.session_state.editing_index    = None
    st.session_state.editing_text     = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Chat","Review dashboard","Eval dashboard"],
                         label_visibility="collapsed")

with st.sidebar:
    st.markdown("---")
    col_a, col_b = st.columns([3,1])
    with col_a:
        st.markdown("**Conversations**")
    with col_b:
        if st.button("New", use_container_width=True):
            if st.session_state.messages:
                save_conversation(st.session_state.conv_id,
                                 get_conv_title(st.session_state.messages),
                                 st.session_state.messages)
            st.session_state.conv_id          = new_conv_id()
            st.session_state.messages         = []
            st.session_state.log_ids          = []
            st.session_state.pending_query    = None
            st.session_state.hitl_state       = None
            st.session_state.active_filter    = None
            st.session_state.run_query        = None
            st.session_state.editing_index    = None
            st.session_state.editing_text     = None
            st.rerun()

    for conv_id, title, updated_at in get_all_conversations():
        is_active = conv_id == st.session_state.conv_id
        col1, col2 = st.columns([4,1])
        with col1:
            label = f"{'→ ' if is_active else ''}{title}"
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                if not is_active:
                    switch_conversation(conv_id)
                    st.rerun()
        with col2:
            if st.button("x", key=f"del_{conv_id}"):
                delete_conversation(conv_id)
                if is_active:
                    st.session_state.conv_id  = new_conv_id()
                    st.session_state.messages = []
                    st.session_state.log_ids  = []
                st.rerun()

    st.markdown("---")
    st.header("Settings")
    top_k        = st.slider("Top chunks", 1, 10, 5)
    show_sources = st.toggle("Show sources",       value=True)
    show_scores  = st.toggle("Show rerank scores", value=False)
    show_latency = st.toggle("Show latency",       value=False)

    st.markdown("---")
    st.markdown("**Example questions**")
    for ex in [
        "What are symptoms of myocardial infarction?",
        "How is type 2 diabetes managed?",
        "What medications treat hypertension?",
        "Describe common knee surgery procedures",
        "What are signs of pneumonia?",
    ]:
        if st.button(ex, use_container_width=True):
            st.session_state.run_query = ex
            st.rerun()

# ── Render assistant message ──────────────────────────────────────────────────
def render_assistant(msg, log_id):
    st.markdown(msg["content"])
    if show_latency and "latency" in msg:
        l = msg["latency"]
        st.markdown(
            f"<div class='latency-text'>Retrieval {l['r']:.0f}ms | "
            f"Rerank {l['rk']:.0f}ms | LLM {l['l']:.0f}ms | "
            f"Total {l['t']:.0f}ms</div>", unsafe_allow_html=True)
    if msg.get("low_conf"):
        st.markdown(
            "<div class='warning-box'>Low confidence — please verify with a clinical source.</div>",
            unsafe_allow_html=True)
    if show_sources and "sources" in msg:
        with st.expander(f"Sources ({len(msg['sources'])} chunks)"):
            for chunk in msg["sources"]:
                sp      = chunk["metadata"].get("specialty","Unknown")
                score   = chunk.get("rerank_score",0)
                text    = chunk["chunk"]
                preview = html.escape(text[:300]) + ("..." if len(text)>300 else "")
                score_html = f"<span class='score-text'>score: {score:.3f}</span>" if show_scores else ""
                st.markdown(
                    f"<div class='source-card'>"
                    f"<span class='specialty-badge'>{html.escape(sp)}</span>"
                    f"{score_html}"
                    f"<div style='margin-top:4px'>{preview}</div>"
                    f"</div>", unsafe_allow_html=True)
    if log_id:
        fb_key = f"fb_{log_id}"
        if fb_key not in st.session_state:
            c1, c2, _ = st.columns([1,1,8])
            with c1:
                if st.button("Good", key=f"up_{log_id}"):
                    save_feedback(log_id, 1)
                    st.session_state[fb_key] = "pos"
                    st.rerun()
            with c2:
                if st.button("Bad", key=f"dn_{log_id}"):
                    st.session_state[fb_key] = "neg_p"
                    st.rerun()
        elif st.session_state[fb_key] == "pos":
            st.caption("Thanks for the feedback!")
        elif st.session_state[fb_key] == "neg_p":
            corr = st.text_area("What was wrong? (optional)", key=f"corr_{log_id}")
            if st.button("Submit", key=f"sub_{log_id}"):
                save_feedback(log_id, 0, corr)
                st.session_state[fb_key] = "neg_d"
                st.rerun()
        elif st.session_state[fb_key] == "neg_d":
            st.caption("Feedback saved. Thank you.")

# ── Execute RAG and append messages ──────────────────────────────────────────
def execute_query(query, specialty_filter):
    session_hist = st.session_state.messages.copy()
    (answer, top_chunks, q_emb,
     r_ms, rk_ms, l_ms, tot_ms,
     top_score, low_conf,
     st_used, lt_used) = run_rag(
        query, vectorstore, bm25, chunks, metadatas,
        reranker_model, groq_client, emb_model,
        top_k, specialty_filter, session_hist)

    log_id = log_response(query, answer, top_chunks,
                          r_ms, rk_ms, l_ms, tot_ms,
                          top_score, specialty_filter, q_emb)
    st.session_state.log_ids.append(log_id)
    logger.info(f"Query='{query}' filter={specialty_filter} score={top_score:.3f} total={tot_ms:.0f}ms")

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": top_chunks, "low_conf": low_conf,
        "latency": {"r": r_ms, "rk": rk_ms, "l": l_ms, "t": tot_ms},
        "st_used": st_used, "lt_used": lt_used,
    })
    save_conversation(st.session_state.conv_id,
                      get_conv_title(st.session_state.messages),
                      st.session_state.messages)
    st.session_state.active_filter = specialty_filter

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Chat
# ══════════════════════════════════════════════════════════════════════════════
if page == "Chat":
    st.title("Clinical Notes Q&A")
    st.caption(
        "Ask questions about clinical cases across 40+ medical specialties including "
        "Cardiology, Neurology, Orthopedics, Gastroenterology, Psychiatry, Ophthalmology, "
        "Urology, Obstetrics & Gynecology, Radiology, and more. "
        "Answers are grounded in 5,000+ real medical transcriptions — never made up."
    )

    # ── 1. Render chat history with inline editing ────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                if st.session_state.editing_index == i:
                    # ── Inline edit box replaces message text ─────────────────
                    new_text = st.text_area(
                        "Edit your message:",
                        value=st.session_state.editing_text,
                        key="edit_area",
                        height=80
                    )
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("Resubmit", type="primary", key=f"resub_{i}"):
                            st.session_state.messages      = st.session_state.messages[:i]
                            st.session_state.log_ids       = st.session_state.log_ids[:i//2]
                            st.session_state.active_filter = None
                            st.session_state.editing_index = None
                            st.session_state.editing_text  = None
                            st.session_state.run_query     = new_text
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_{i}"):
                            st.session_state.editing_index = None
                            st.session_state.editing_text  = None
                            st.rerun()
                else:
                    # ── Normal message display with Edit button ────────────────
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.markdown(msg["content"])
                    with col2:
                        if st.button("Edit", key=f"edit_{i}"):
                            st.session_state.editing_index = i
                            st.session_state.editing_text  = msg["content"]
                            st.rerun()
            else:
                idx    = i // 2
                log_id = st.session_state.log_ids[idx] \
                         if idx < len(st.session_state.log_ids) else None
                render_assistant(msg, log_id)

    # ── 2. HITL panel ─────────────────────────────────────────────────────────
    if st.session_state.hitl_state == "waiting":
        query     = st.session_state.pending_query
        auto_sp   = detect_specialty(query)
        prev_sp   = st.session_state.active_filter or []
        suggested = list(dict.fromkeys(auto_sp + prev_sp))
        prev_user = [m for m in st.session_state.messages if m["role"] == "user"]

        with st.container():
            st.markdown("<div class='hitl-box'>", unsafe_allow_html=True)
            st.markdown(f"**Your question:** {query}")
            if prev_user:
                st.caption(f"Previous topic: \"{prev_user[-1]['content'][:60]}\"")
            st.markdown("Which specialty should I focus on?")
            selected = st.multiselect("Select specialty (leave empty = search all)",
                                      ALL_SPECIALTIES, default=suggested)
            custom   = st.text_input("Or type a specialty not in the list:")
            col1, col2 = st.columns([1,3])
            with col1:
                if st.button("Search", type="primary"):
                    final = selected + ([custom.strip()] if custom.strip() else [])
                    st.session_state.specialty_filter = final if final else None
                    st.session_state.hitl_state       = "confirmed"
                    st.rerun()
            with col2:
                if st.button("Search all specialties"):
                    st.session_state.specialty_filter = None
                    st.session_state.hitl_state       = "confirmed"
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Execute confirmed HITL ─────────────────────────────────────────────
    elif st.session_state.hitl_state == "confirmed":
        query = st.session_state.pending_query
        sf    = st.session_state.specialty_filter
        with st.spinner(f"Searching {', '.join(sf) if sf else 'all specialties'}..."):
            try:
                execute_query(query, sf)
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                st.error(f"Something went wrong: {e}")
        st.session_state.hitl_state       = None
        st.session_state.pending_query    = None
        st.session_state.specialty_filter = None
        st.rerun()

    # ── 4. Execute queued query ───────────────────────────────────────────────
    elif st.session_state.run_query is not None:
        query = st.session_state.run_query
        st.session_state.run_query = None

        has_history = len(st.session_state.messages) > 0

        if is_followup(query, has_history):
            sf = st.session_state.active_filter
        elif not is_medical_query(query):
            st.warning("This assistant is for clinical and medical questions only.")
            sf    = None
            query = None
        else:
            auto_sp = detect_specialty(query)
            sf      = auto_sp if auto_sp else None
            if not auto_sp:
                st.session_state.pending_query = query
                st.session_state.hitl_state    = "waiting"
                st.rerun()
                query = None

        if query:
            label = f"Searching {', '.join(sf)}..." if sf else "Retrieving answer..."
            with st.spinner(label):
                try:
                    execute_query(query, sf)
                except Exception as e:
                    logger.error(f"Pipeline error: {e}", exc_info=True)
                    st.error(f"Something went wrong: {e}")
            st.rerun()

    # ── 5. Chat input — always rendered last ──────────────────────────────────
    if st.session_state.editing_index is None:
        user_input = st.chat_input("Ask a clinical question...")
        if user_input:
            st.session_state.run_query = user_input
            st.rerun()

    if st.session_state.messages:
        if st.button("Clear chat"):
            st.session_state.messages      = []
            st.session_state.log_ids       = []
            st.session_state.pending_query = None
            st.session_state.hitl_state    = None
            st.session_state.active_filter = None
            st.session_state.run_query     = None
            st.session_state.editing_index = None
            st.session_state.editing_text  = None
            st.session_state.conv_id       = new_conv_id()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Review dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Review dashboard":
    st.title("Review dashboard")
    st.caption("All logged queries, answers, latency, and feedback.")

    logs = get_all_logs()
    if not logs:
        st.info("No queries logged yet.")
        st.stop()

    total     = len(logs)
    positive  = sum(1 for l in logs if l["feedback"] == 1)
    negative  = sum(1 for l in logs if l["feedback"] == 0)
    avg_lat   = sum(l["total_ms"] or 0 for l in logs) / total
    avg_score = sum(l["top_rerank_score"] or 0 for l in logs) / total

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total queries",     total)
    c2.metric("Positive feedback", positive)
    c3.metric("Negative feedback", negative)
    c4.metric("Avg latency",       f"{avg_lat:.0f}ms")
    c5.metric("Avg rerank score",  f"{avg_score:.3f}")

    st.divider()
    filter_fb = st.selectbox("Filter", ["All","Positive","Negative","No feedback"])
    filtered  = {
        "All":         logs,
        "Positive":    [l for l in logs if l["feedback"] == 1],
        "Negative":    [l for l in logs if l["feedback"] == 0],
        "No feedback": [l for l in logs if l["feedback"] is None],
    }[filter_fb]

    st.caption(f"Showing {len(filtered)} of {total} entries")
    for log in filtered:
        fb_label = {1:"Good",0:"Bad",None:"no feedback"}.get(log["feedback"],"no feedback")
        with st.expander(f"{log['timestamp'][:16]}  |  {log['query'][:80]}  |  {fb_label}"):
            st.markdown(f"**Query:** {log['query']}")
            st.markdown(f"**Specialty filter:** {log.get('specialty_filter','none')}")
            st.markdown(f"**Answer:** {log['answer']}")
            st.markdown(f"**Specialties:** {log['specialties']}")
            st.markdown(
                f"**Latency:** retrieval {log['retrieval_ms'] or 0:.0f}ms | "
                f"rerank {log['rerank_ms'] or 0:.0f}ms | "
                f"LLM {log['llm_ms'] or 0:.0f}ms | "
                f"total {log['total_ms'] or 0:.0f}ms")
            st.markdown(f"**Top rerank score:** {log['top_rerank_score'] or 0:.3f}")
            if log.get("correction"):
                st.markdown(f"**User correction:** {log['correction']}")
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Eval dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Eval dashboard":
    import pandas as pd
    import json

    st.title("Eval dashboard")
    st.caption("RAGAs evaluation results — Faithfulness, Answer Relevancy, Context Recall")

    csv_path = "eval_results.csv"
    if not os.path.exists(csv_path):
        st.info("No eval results found. Run eval.py first to generate eval_results.csv")
        st.code("conda activate rag-eval\npython eval.py")
        st.stop()

    df = pd.read_csv(csv_path)

    metric_cols = [c for c in ["faithfulness","answer_relevancy","context_recall"]
                   if c in df.columns]

    if not metric_cols:
        st.error("No metric columns found in eval_results.csv")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    cols = st.columns(len(metric_cols))
    for col, metric in zip(cols, metric_cols):
        score = df[metric].mean()
        label = metric.replace("_", " ").title()
        delta_color = "normal"
        if score >= 0.8:
            delta = "Good"
        elif score >= 0.6:
            delta = "OK"
        else:
            delta = "Needs work"
        col.metric(label, f"{score:.3f}", delta)

    st.divider()

    # ── Bar chart — per question ──────────────────────────────────────────────
    st.subheader("Per-question scores")

    chart_df = df[["question"] + metric_cols].copy()
    chart_df["question"] = chart_df["question"].str[:50]
    chart_df = chart_df.set_index("question")

    st.bar_chart(chart_df)

    # ── Aggregate score bars ──────────────────────────────────────────────────
    st.subheader("Aggregate scores")

    for metric in metric_cols:
        score = df[metric].mean()
        label = metric.replace("_", " ").title()
        if score >= 0.8:
            color = "green"
        elif score >= 0.6:
            color = "orange"
        else:
            color = "red"
        filled   = int(score * 20)
        empty    = 20 - filled
        bar      = "█" * filled + "░" * empty
        st.markdown(f"**{label}** `{bar}` **{score:.3f}**")

    # ── Worst performing questions ────────────────────────────────────────────
    st.divider()
    st.subheader("Lowest scoring questions")
    st.caption("Questions where at least one metric scored below 0.6")

    df["avg_score"] = df[metric_cols].mean(axis=1)
    weak = df[df["avg_score"] < 0.6].sort_values("avg_score")

    if weak.empty:
        st.success("All questions scored above 0.6 average — system is performing well.")
    else:
        for _, row in weak.iterrows():
            with st.expander(f"{row['question'][:80]} — avg: {row['avg_score']:.3f}"):
                for metric in metric_cols:
                    score = row[metric]
                    st.markdown(f"**{metric}:** {score:.3f}")

    # ── Raw table ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full results table")
    display_cols = ["question"] + metric_cols + ["avg_score"]
    st.dataframe(
        df[display_cols].style.format(
            {m: "{:.3f}" for m in metric_cols + ["avg_score"]}
        ).background_gradient(
            subset=metric_cols, cmap="RdYlGn", vmin=0, vmax=1
        ),
        use_container_width=True
    )