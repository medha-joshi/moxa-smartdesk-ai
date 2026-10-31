"""
tools/rag.py — Hybrid RAG pipeline (dense + BM25 sparse in Qdrant, cross-encoder re-ranking).

Each knowledge base (HR, IT) gets its own Qdrant collection holding two vectors per chunk:
  * dense  — OpenAI embeddings, cosine similarity
  * sparse — BM25 term weights (IDF applied by Qdrant)
A query runs both searches, fuses them with Reciprocal Rank Fusion, then a cross-encoder
re-ranks the fused candidates. The re-rank score decides whether the answer is confident.
"""

import atexit
import json
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from smartDeskAi.config import (
    DATA_DIR, QDRANT_API_KEY, QDRANT_PATH, QDRANT_URL,
    RERANK_MODEL, RERANK_SCORE_THRESHOLD, SPARSE_MODEL, embeddings,
)
from smartDeskAi.logger import setup_logger

logger = setup_logger("smartDeskAi.tools.rag")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
CANDIDATES_PER_RETRIEVER = 10
TOP_K = 3

KB_FILES = {"hr": "hr_kb.json", "it": "it_kb.json"}

# The embedded (on-disk) Qdrant client is not thread-safe, and HR/IT agents search in parallel
_qdrant_lock = threading.Lock()


@dataclass
class RetrievedChunk:
    id: str
    category: str
    content: str
    rerank_score: float


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    # One shared client: embedded Qdrant locks its storage folder per client instance
    client = QdrantClient(path=QDRANT_PATH)
    atexit.register(client.close)
    return client


@lru_cache(maxsize=1)
def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=SPARSE_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=RERANK_MODEL)


def load_documents(kb_name: str) -> list[Document]:
    """Turn each Q&A entry of a knowledge-base JSON file into one chunk."""
    with open(DATA_DIR / KB_FILES[kb_name], encoding="utf-8") as f:
        entries = json.load(f)
    documents = []
    for entry in entries:
        category = " / ".join(filter(None, [entry.get("category"), entry.get("subcategory")]))
        documents.append(Document(
            page_content=f"Question: {entry['question']}\nAnswer: {entry['answer']}",
            metadata={"kb_id": entry["id"], "category": category},
        ))
    return documents


class KnowledgeBase:
    """A searchable HR or IT knowledge base backed by one Qdrant collection."""

    def __init__(self, kb_name: str, rebuild: bool = False):
        self.kb_name = kb_name
        self.collection_name = f"smartdesk_{kb_name}_kb"
        self.client = get_qdrant_client()
        self.sparse_embeddings = get_sparse_embeddings()
        self.reranker = get_reranker()
        self._ensure_index(rebuild)

    def _ensure_index(self, rebuild: bool) -> None:
        documents = load_documents(self.kb_name)
        with _qdrant_lock:
            exists = self.client.collection_exists(self.collection_name)
            if exists and not rebuild and self.client.count(self.collection_name).count == len(documents):
                logger.info("Using existing Qdrant collection %s", self.collection_name)
                return
            if exists:
                self.client.delete_collection(self.collection_name)
            dense_size = len(embeddings.embed_query("dimension probe"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name=DENSE_VECTOR_NAME,
                sparse_vector_name=SPARSE_VECTOR_NAME,
            )
            # Deterministic ids keep re-indexing idempotent
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, doc.metadata["kb_id"])) for doc in documents]
            vector_store.add_documents(documents, ids=ids)
        logger.info("Indexed %d chunks into %s", len(documents), self.collection_name)

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Hybrid search + re-rank. Returns only chunks whose re-rank score passes the confidence threshold."""
        dense_vector = embeddings.embed_query(query)
        sparse = self.sparse_embeddings.embed_query(query)
        sparse_vector = models.SparseVector(indices=sparse.indices, values=sparse.values)

        with _qdrant_lock:
            points = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME, limit=CANDIDATES_PER_RETRIEVER),
                    models.Prefetch(query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=CANDIDATES_PER_RETRIEVER),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=CANDIDATES_PER_RETRIEVER,
                with_payload=True,
            ).points

        if not points:
            logger.info("[%s] no chunks retrieved for %r", self.kb_name, query)
            return []

        contents = [point.payload["page_content"] for point in points]
        scores = list(self.reranker.rerank(query, contents))
        ranked = sorted(zip(points, scores), key=lambda pair: pair[1], reverse=True)

        chunks = [
            RetrievedChunk(
                id=point.payload["metadata"]["kb_id"],
                category=point.payload["metadata"]["category"],
                content=point.payload["page_content"],
                rerank_score=float(score),
            )
            for point, score in ranked[:top_k]
        ]
        logger.info("[%s] %r -> %s", self.kb_name, query, [(c.id, round(c.rerank_score, 2)) for c in chunks])
        return [chunk for chunk in chunks if chunk.rerank_score >= RERANK_SCORE_THRESHOLD]


_kb_lock = threading.Lock()


@lru_cache(maxsize=None)
def _cached_knowledge_base(kb_name: str) -> KnowledgeBase:
    return KnowledgeBase(kb_name)


def get_knowledge_base(kb_name: str) -> KnowledgeBase:
    # Serialize first-time construction so parallel agents don't index the same collection twice
    with _kb_lock:
        return _cached_knowledge_base(kb_name)


def rebuild_knowledge_bases() -> None:
    """Re-index every knowledge base from its JSON file."""
    with _kb_lock:
        _cached_knowledge_base.cache_clear()
        for kb_name in KB_FILES:
            KnowledgeBase(kb_name, rebuild=True)


def format_search_results(kb_label: str, query: str, chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks for the LLM, or an explicit no-confident-match signal."""
    if not chunks:
        return (
            f"NO_CONFIDENT_MATCH: The {kb_label} knowledge base has no entry that confidently answers "
            f"'{query}'. Do NOT answer from general knowledge. Tell the employee you don't have enough "
            "information on this and offer to create a support ticket."
        )
    blocks = [
        f"[{i}] (source: {chunk.id} | {chunk.category} | relevance {chunk.rerank_score:.2f})\n{chunk.content}"
        for i, chunk in enumerate(chunks, start=1)
    ]
    return (
        f"{len(chunks)} relevant {kb_label} knowledge-base entries found. Answer ONLY from these; "
        "if they do not actually answer the question, say you don't have enough information.\n\n"
        + "\n\n".join(blocks)
    )
