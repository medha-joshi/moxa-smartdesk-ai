"""
Shared test setup.

The environment is fixed *before* any smartDeskAi module is imported: smartDeskAi.config reads it at import
time, and load_dotenv() never overrides variables that are already set. So the real .env is ignored and
no test can reach OpenAI, Jira, LangSmith or the on-disk Qdrant data.
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="moxa-tests-"))

os.environ.update({
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_API_KEY": "",
    "JIRA_SERVER": "https://jira.example.test",
    "JIRA_EMAIL": "bot@example.test",
    "JIRA_API_TOKEN": "test-token",
    "JIRA_PROJECT_KEY": "IKCS",
    "QDRANT_URL": "",
    "QDRANT_PATH": str(_TMP / "qdrant"),
})

# Keep test runs out of log/app.log
from smartDeskAi import logger as _logger  # noqa: E402

_logger.LOGGING_CONFIG["handlers"]["file"]["filename"] = str(_TMP / "test.log")
_logger.LOG_DIR = _TMP

import pytest  # noqa: E402

from smartDeskAi.tools.rag import RetrievedChunk  # noqa: E402


class FakeKnowledgeBase:
    """Stands in for rag.KnowledgeBase: returns canned chunks and records the queries."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = chunks or []
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks[:top_k]


@pytest.fixture
def hr_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="hr_kb0001",
        category="Workplace Policies & Conduct",
        content="Question: What is the dress code at the office?\n"
                "Answer: Synergy Inc follows a business casual dress code. Casual attire is allowed on Fridays.",
        rerank_score=3.2,
    )


@pytest.fixture
def it_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="it_kb0001",
        category="Access & Accounts / Password Resets",
        content="Question: How do I reset my password?\n"
                "Answer: Use the self-service password portal and verify with MFA.",
        rerank_score=2.5,
    )
