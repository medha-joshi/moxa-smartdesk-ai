"""
tools/kb_phrasings.py — Generate alternate phrasings for knowledge-base entries.

    uv run moxa-kb-phrasings [--kb hr|it] [--count 5] [--overwrite]

Employees rarely ask in the KB's wording ("Can I wear jeans?" vs "What is the dress code?"), and
paraphrased questions are where retrieval is weakest. For each entry the LLM writes a few questions an
employee might ask instead, stored as `alt_questions` in the KB JSON and indexed with the entry (see
rag.load_documents). The LLM sees only the KB entry, never the evaluation datasets, so the retrieval
eval still measures generalisation. Entries that already have phrasings are kept unless --overwrite.
Review the generated phrasings before committing them, like any other KB content.
"""

import argparse
import json

from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, Field

from smartDeskAi.config import DATA_DIR, llm
from smartDeskAi.tools.rag import KB_FILES

PROMPT = """You help a company help-desk search engine match employee questions to knowledge-base entries.

Knowledge-base entry
Category: {category}
Question: {question}
Answer: {answer}

Write {count} different questions that an employee might type when this entry is the answer they need.
- Write as real employees do: short, casual, first person, sometimes describing a situation or symptom
  ("I'm locked out", "my screen is black") rather than naming the policy.
- Vary the vocabulary: use everyday words and synonyms instead of the entry's own terms.
- Each one must be answered by this entry; don't drift into neighbouring topics the answer doesn't cover.
- No numbering, no duplicates."""


class Phrasings(BaseModel):
    questions: list[str] = Field(description="Alternate employee questions answered by the entry")


def generate(entry: dict, count: int) -> list[str]:
    result = llm.with_structured_output(Phrasings).invoke(PROMPT.format(count=count, **{
        "category": " / ".join(filter(None, [entry.get("category"), entry.get("subcategory")])),
        "question": entry["question"], "answer": entry["answer"],
    }))
    unique = {entry["question"].strip().lower(): None}  # drop repeats of the KB question and of each other
    for question in map(str.strip, result.questions):
        if question:
            unique.setdefault(question.lower(), question)
    return [q for q in unique.values() if q]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alternate phrasings for KB entries.")
    parser.add_argument("--kb", choices=sorted(KB_FILES), action="append", help="knowledge base(s); default: all")
    parser.add_argument("--count", type=int, default=5, help="phrasings per entry")
    parser.add_argument("--overwrite", action="store_true", help="regenerate entries that already have phrasings")
    args = parser.parse_args()

    for kb in args.kb or KB_FILES:
        path = DATA_DIR / KB_FILES[kb]
        entries = json.loads(path.read_text(encoding="utf-8"))
        todo = [e for e in entries if args.overwrite or not e.get("alt_questions")]
        print(f"{kb}: generating phrasings for {len(todo)} of {len(entries)} entries")
        if not todo:
            continue
        results = RunnableLambda(lambda e: generate(e, args.count)).batch(todo, config=RunnableConfig(max_concurrency=8))
        for entry, phrasings in zip(todo, results):
            entry["alt_questions"] = phrasings
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{kb}: wrote {path}")


if __name__ == "__main__":
    main()
