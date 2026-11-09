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
# LLM-as-judge for the answer evaluation (faithfulness / correctness); deterministic so reruns agree
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL)
judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0, max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT)

# ---------- Embeddings (for RAG) ----------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, max_retries=LLM_MAX_RETRIES)

# ---------- Qdrant / retrieval ----------
# QDRANT_URL selects a Qdrant server; otherwise an embedded on-disk Qdrant at QDRANT_PATH is used
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_PATH = os.environ.get("QDRANT_PATH", str(PROJECT_ROOT / ".qdrant_data"))
SPARSE_MODEL = "Qdrant/bm25"
# bge-reranker-v2-m3 (int8) handles paraphrased questions far better than ms-marco-MiniLM-L-6-v2:
# on the retrieval eval it separates answerable from out-of-scope queries much more cleanly
# (~89% vs ~68% pipeline accuracy with every out-of-scope query rejected), at ~0.5s vs ~0.05s per query.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8")
# Cross-encoder logit below which a chunk is not a confident match. The scale depends on RERANK_MODEL,
# so re-tune it with `moxa-eval-retrieval` (threshold sweep) whenever the model changes. For
# bge-reranker-v2-m3 on the 107-query retrieval eval, -5.0 answers 94% of answerable questions but lets
# 25% of out-of-scope near-misses through ("maternity leave" vs the leave allowance score up to about -4).
# Those are left to the agent's self-assessment, which declined all 10 such near-misses in the answer eval
# with gpt-4o; -3.0 would reject 95% of them at retrieval but also drop 2 answerable questions.
RERANK_SCORE_THRESHOLD = float(os.environ.get("RERANK_SCORE_THRESHOLD", "-5.0"))

# ---------- Jira ----------
JIRA_SERVER = os.environ.get("JIRA_SERVER", "").rstrip("/")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN", "")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "IKCS")
JIRA_TIMEOUT = float(os.environ.get("JIRA_TIMEOUT", "15"))
