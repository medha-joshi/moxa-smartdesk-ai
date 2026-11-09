import json

import pytest

from smartDeskAi.config import DATA_DIR
from smartDeskAi.tools import hr_tools, it_tools, rag
from smartDeskAi.tools.rag import (
    KB_FILES, KnowledgeBase, RetrievedChunk, add_citations, ensure_citations, format_search_results, load_documents,
)
from tests.conftest import FakeKnowledgeBase


# ---------- knowledge-base files ----------

@pytest.mark.parametrize("kb_name", KB_FILES)
def test_kb_files_are_well_formed(kb_name):
    entries = json.loads((DATA_DIR / KB_FILES[kb_name]).read_text(encoding="utf-8"))
    ids = [e["id"] for e in entries]
    assert entries and len(ids) == len(set(ids)), "KB ids must be unique"
    for entry in entries:
        assert entry["id"].startswith(f"{kb_name}_kb")
        assert entry["question"].strip() and entry["answer"].strip() and entry["category"].strip()


def test_load_documents_includes_alt_questions_and_metadata():
    doc = next(d for d in load_documents("hr") if d.metadata["kb_id"] == "hr_kb0001")
    assert doc.page_content.startswith("Question: What is the dress code at the office?\nAlso asked as: ")
    assert "\nAnswer: Synergy Inc follows a business casual dress code." in doc.page_content
    assert doc.metadata["category"] == "Workplace Policies & Conduct"


def test_category_joins_subcategory():
    doc = next(d for d in load_documents("it") if d.metadata["kb_id"] == "it_kb0001")
    assert doc.metadata["category"] == "Access & Accounts / Password Resets"


# ---------- citations ----------

def test_add_citations_numbers_markers_and_lists_sources():
    text = "Business casual applies [hr_kb0001]. Reset via the portal [it_kb0001, hr_kb0001]."
    result = add_citations(text)
    body, sources = result.split("\n\nSources:\n")
    assert body == "Business casual applies [1]. Reset via the portal [1][2]."
    assert sources.splitlines() == [
        '- [1] HR knowledge base · Workplace Policies & Conduct: "What is the dress code at the office?" (hr_kb0001)',
        sources.splitlines()[1],
    ]
    assert sources.splitlines()[1].startswith("- [2] IT knowledge base · Access & Accounts / Password Resets:")


def test_add_citations_drops_made_up_ids():
    assert add_citations("Something [hr_kb9999].") == "Something."


def test_add_citations_without_markers_is_unchanged():
    assert add_citations("Hello! How can I help?") == "Hello! How can I help?"


def search_result(*chunks: RetrievedChunk) -> str:
    return format_search_results("HR", "dress code", list(chunks))


def test_ensure_citations_adds_marker_for_used_entry(hr_chunk, it_chunk):
    reply = "Synergy Inc follows a business casual dress code, and casual attire is allowed on Fridays."
    assert ensure_citations(reply, [search_result(hr_chunk, it_chunk)]) == f"{reply} [hr_kb0001]"


def test_ensure_citations_leaves_cited_unrelated_or_declined_replies(hr_chunk):
    results = [search_result(hr_chunk)]
    cited = "Business casual [hr_kb0001]."
    assert ensure_citations(cited, results) == cited
    unrelated = "Please contact the service desk for anything else."
    assert ensure_citations(unrelated, results) == unrelated
    declined = "Sorry, I don't have enough information on that business casual dress code question."
    assert ensure_citations(declined, results) == declined


def test_format_search_results(hr_chunk, it_chunk):
    result = format_search_results("HR", "dress code", [hr_chunk, it_chunk])
    assert result.startswith("2 relevant HR knowledge-base entries found.")
    assert "[1] (source: hr_kb0001 | Workplace Policies & Conduct | relevance 3.20)" in result
    assert "[2] (source: it_kb0001" in result
    assert "e.g. [hr_kb0001]" in result


def test_format_search_results_no_match():
    assert format_search_results("IT", "pizza", []).startswith("NO_CONFIDENT_MATCH")


# ---------- search threshold ----------

def test_search_keeps_top_k_above_threshold(monkeypatch):
    monkeypatch.setattr(rag, "RERANK_SCORE_THRESHOLD", 0.0)
    kb = object.__new__(KnowledgeBase)  # skip Qdrant / model loading
    kb.kb_name = "hr"
    ranked = [RetrievedChunk(f"hr_kb000{i}", "c", "x", score) for i, score in enumerate([5.0, 1.0, -2.0, 4.0])]
    kb.rank = lambda query: ([c.id for c in ranked], ranked)
    assert [c.id for c in kb.search("q", top_k=3)] == ["hr_kb0000", "hr_kb0001"]
    kb.rank = lambda query: ([], [])
    assert kb.search("q") == []


# ---------- search tools ----------

@pytest.mark.parametrize("module, tool, kb_name, label", [
    (hr_tools, hr_tools.search_hr_knowledge_base, "hr", "HR"),
    (it_tools, it_tools.search_it_knowledge_base, "it", "IT"),
])
def test_search_tools(monkeypatch, hr_chunk, module, tool, kb_name, label):
    fake = FakeKnowledgeBase([hr_chunk])
    requested = []
    monkeypatch.setattr(module, "get_knowledge_base", lambda name: requested.append(name) or fake)

    result = tool.invoke({"query": "What is the dress code?"})

    assert requested == [kb_name] and fake.queries == ["What is the dress code?"]
    assert result.startswith(f"1 relevant {label} knowledge-base entries found.")


@pytest.mark.parametrize("module, tool", [
    (hr_tools, hr_tools.search_hr_knowledge_base), (it_tools, it_tools.search_it_knowledge_base),
])
def test_search_tools_report_outage(monkeypatch, module, tool):
    def broken(name):
        raise ConnectionError("qdrant down")
    monkeypatch.setattr(module, "get_knowledge_base", broken)
    assert "temporarily unavailable" in tool.invoke({"query": "anything"})
