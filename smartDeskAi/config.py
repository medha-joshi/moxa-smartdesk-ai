"""
config.py — LLM, embeddings, and OpenAI client setup for smartDeskAiAssitant
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in your .env file or environment. Set OPENAI_API_KEY in your .env file")

# ---------- OpenAI client ----------
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- LangChain LLM ----------
# max_retries gives exponential-backoff retries on transient OpenAI errors (rate limits, timeouts, 5xx)
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT)

# ---------- Embeddings (for RAG) ----------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, max_retries=LLM_MAX_RETRIES)

# ---------- Qdrant / retrieval ----------
# QDRANT_URL selects a Qdrant server; otherwise an embedded on-disk Qdrant at QDRANT_PATH is used
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_PATH = os.environ.get("QDRANT_PATH", str(PROJECT_ROOT / ".qdrant_data"))
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# Cross-encoder logit below which a chunk is not a confident match. Calibrated on sample queries:
# direct hits >= 0.8, paraphrased hits down to about -2.5, unanswerable questions <= -5.5.
# Borderline chunks that pass are still checked by the LLM's self-assessment.
RERANK_SCORE_THRESHOLD = float(os.environ.get("RERANK_SCORE_THRESHOLD", "-4.0"))

# ---------- Jira ----------
JIRA_SERVER = os.environ.get("JIRA_SERVER", "").rstrip("/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "IKCS")
JIRA_TIMEOUT = float(os.environ.get("JIRA_TIMEOUT", "15"))
