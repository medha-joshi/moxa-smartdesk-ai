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
import os
import re
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
# Only the best fused candidates are re-ranked: the cross-encoder is the slowest step, and with the KB's
# alternate phrasings the fused ranking already puts the right entry in its top 2 on the retrieval eval.
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "5"))
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
    """Turn each Q&A entry of a knowledge-base JSON file into one chunk.

    An entry's alternate phrasings (`alt_questions`, see kb_phrasings.py) go into the chunk too, so dense
    search, BM25 and the re-ranker all match employees who don't use the KB's own wording.
    """
    with open(DATA_DIR / KB_FILES[kb_name], encoding="utf-8") as f:
        entries = json.load(f)
    documents = []
    for entry in entries:
        category = " / ".join(filter(None, [entry.get("category"), entry.get("subcategory")]))
        also_asked = f"\nAlso asked as: {' | '.join(entry['alt_questions'])}" if entry.get("alt_questions") else ""
        documents.append(Document(
            page_content=f"Question: {entry['question']}{also_asked}\nAnswer: {entry['answer']}",
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
            if exists and not rebuild and self._indexed_contents() == {doc.page_content for doc in documents}:
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

    def _indexed_contents(self) -> set[str]:
        """Every chunk text in the collection, to detect KB edits (the KBs are small enough to read whole)."""
        points, _ = self.client.scroll(self.collection_name, limit=10_000, with_payload=["page_content"])
        return {point.payload["page_content"] for point in points}

    def embed(self, query: str) -> tuple[list[float], models.SparseVector]:
        """The query's dense and sparse (BM25) vectors."""
        sparse = self.sparse_embeddings.embed_query(query)
        return embeddings.embed_query(query), models.SparseVector(indices=sparse.indices, values=sparse.values)

    def single_retriever(self, vector: list[float] | models.SparseVector, using: str) -> list[str]:
        """KB ids from one retriever alone (dense or sparse), best first. Used by the retrieval evaluation."""
        with _qdrant_lock:
            points = self.client.query_points(
                collection_name=self.collection_name, query=vector, using=using,
                limit=CANDIDATES_PER_RETRIEVER, with_payload=True,
            ).points
        return [point.payload["metadata"]["kb_id"] for point in points]

    def rank(
        self, query: str, vectors: tuple[list[float], models.SparseVector] | None = None,
    ) -> tuple[list[str], list[RetrievedChunk]]:
        """Hybrid search + re-rank, before the top-k cut and the confidence threshold.

        Returns the fused (RRF) candidate ids in fusion order, and the top RERANK_CANDIDATES of them
        re-ranked best first.
        """
        dense_vector, sparse_vector = vectors or self.embed(query)
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
            return [], []

        # Re-rank against the full "Question: … Answer: …" text. Scoring the KB question alone was tried
        # and is much worse on paraphrased queries (retrieval eval: 61% vs 88% pipeline accuracy).
        candidates = points[:RERANK_CANDIDATES]
        contents = [point.payload["page_content"] for point in candidates]
        scores = list(self.reranker.rerank(query, contents))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        fused_ids = [point.payload["metadata"]["kb_id"] for point in points]
        return fused_ids, [
            RetrievedChunk(
                id=point.payload["metadata"]["kb_id"],
                category=point.payload["metadata"]["category"],
                content=point.payload["page_content"],
                rerank_score=float(score),
            )
            for point, score in ranked
        ]

    def search(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Hybrid search + re-rank. Returns only chunks whose re-rank score passes the confidence threshold."""
        _, ranked = self.rank(query)
        if not ranked:
            logger.info("[%s] no chunks retrieved for %r", self.kb_name, query)
            return []
        chunks = ranked[:top_k]
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


# An agent's citation marker: [hr_kb0001] or [hr_kb0001, it_kb0004], with any space before it
CITATION = re.compile(r"(\s*)\[\s*((?:hr|it)_kb\d+(?:\s*[,;]\s*(?:hr|it)_kb\d+)*)\s*\]")


@lru_cache(maxsize=1)
def kb_entries() -> dict[str, Document]:
    return {doc.metadata["kb_id"]: doc for kb_name in KB_FILES for doc in load_documents(kb_name)}


def add_citations(text: str) -> str:
    """Turn the agents' [hr_kb0001] markers into numbered citations plus a Sources list.

    IDs that are not in either knowledge base are dropped, so a made-up citation is never shown.
    """
    entries = kb_entries()
    numbers: dict[str, int] = {}

    def number(match: re.Match) -> str:
        refs = [
            numbers.setdefault(kb_id, len(numbers) + 1)
            for kb_id in re.split(r"\s*[,;]\s*", match.group(2))
            if kb_id in entries
        ]
        return match.group(1) + "".join(f"[{n}]" for n in sorted(set(refs))) if refs else ""

    body = CITATION.sub(number, text)
    if not numbers:
        return body
    sources = []
    for kb_id, n in numbers.items():
        doc = entries[kb_id]
        question = doc.page_content.split("\n", 1)[0].removeprefix("Question: ")
        sources.append(f"- [{n}] {kb_id[:2].upper()} knowledge base · {doc.metadata['category']}: \"{question}\" ({kb_id})")
    return f"{body}\n\nSources:\n" + "\n".join(sources)


# One entry of format_search_results() output: its source ID and its "Question: … Answer: …" text
SEARCH_RESULT = re.compile(r"\(source: ((?:hr|it)_kb\d+) \|[^\n]*\n(.*?)(?=\n\n\[\d+\] \(source: |\Z)", re.S)
MIN_WORD_OVERLAP = 0.3


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def ensure_citations(reply: str, search_results: list[str]) -> str:
    """Add citation markers to a reply that used search results but cited none.

    The LLM sometimes drops the markers, typically when it rewrites an entry as a bullet list. Only
    entries whose answer shares enough words with the reply are cited, so an unused hit is never shown.
    """
    if CITATION.search(reply) or "don't have enough information" in reply.lower():
        return reply
    reply_words = _words(reply)
    cited = []
    for kb_id, content in (match for result in search_results for match in SEARCH_RESULT.findall(result)):
        answer_words = _words(content.split("\nAnswer: ", 1)[-1])
        if answer_words and len(answer_words & reply_words) / len(answer_words) >= MIN_WORD_OVERLAP:
            cited.append(kb_id)
    if not cited:
        return reply
    logger.info("Added missing citations %s", cited)
    return f"{reply} [{', '.join(dict.fromkeys(cited))}]"


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
        "if they do not actually answer the question, say you don't have enough information. "
        f"Cite the source ID after each fact you use, e.g. [{chunks[0].id}].\n\n"
        + "\n\n".join(blocks)
    )
