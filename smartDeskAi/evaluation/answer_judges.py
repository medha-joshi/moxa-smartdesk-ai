"""
evaluation/answer_judges.py — LLM-as-judge graders for Moxa's answers.

- Faithfulness: is every factual claim in the answer supported by the knowledge-base results the agents
  retrieved? The judge splits the answer into claims and checks each one against that context only
  (never against general knowledge), so a claim that is true in the world but absent from the KB still
  counts as unsupported. Score = supported claims / claims; an answer with no factual claims (a greeting,
  "I don't have enough information...") scores 1.0, since it asserts nothing.
- Correctness: does the answer give the employee the right information, compared with a reference
  answer written from the knowledge base? correct = 1.0, partially correct = 0.5, incorrect = 0.0.

Verdicts are cached in memory by their inputs, so a LangSmith run and the local report built from it
grade each answer only once.
"""

import threading
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from smartDeskAi.config import judge_llm

FAITHFULNESS_PROMPT = """You grade the faithfulness of answers given by Moxa, an HR & IT help-desk assistant.
Moxa must answer ONLY from the knowledge-base search results it retrieved. You get the employee's question,
those search results (the CONTEXT) and Moxa's answer.

1. List every factual claim the answer makes about the company, its policies, procedures, numbers,
   timelines, tools, portals, teams or contacts. Split compound sentences into separate claims.
   Do NOT list: greetings, empathy, offers to raise a support ticket, questions to the employee, requests
   for their email, statements that Moxa lacks information, or citation markers like [1].
2. For each claim decide whether the CONTEXT supports it. Supported means the context states it or it
   follows directly from it; a faithful paraphrase is supported. Judge against the CONTEXT only, never your
   own knowledge: a plausible or generally true claim that the context does not contain is unsupported.
   If the context is empty or says NO_CONFIDENT_MATCH, every factual claim is unsupported.
"""

CORRECTNESS_PROMPT = """You grade the correctness of answers given by Moxa, an HR & IT help-desk assistant,
against a reference answer written from the company knowledge base. You get the employee's question, the
reference and Moxa's answer.

Verdicts:
- correct: the answer conveys the reference's key facts needed to answer the question, and nothing in it
  contradicts the reference. Wording, order and formatting don't matter. Extra details are fine unless they
  contradict the reference. When the reference says Moxa should NOT answer (no information, out of scope),
  an answer that honestly says so (and, if the reference mentions it, offers a ticket or steers back to
  HR/IT) is correct.
- partially_correct: nothing is wrong, but some key facts of the reference are missing, or the employee
  would need to ask again to get the full answer.
- incorrect: the answer contradicts the reference (wrong numbers, steps, eligibility...), answers with
  invented details when the reference says the information is unavailable, or refuses / says it doesn't
  know when the reference has the answer.
List the key facts that are missing and the statements that are wrong (empty lists if none).
"""


class Claim(BaseModel):
    claim: str = Field(description="One factual claim made in the answer")
    supported: bool = Field(description="Whether the context supports the claim")


class FaithfulnessVerdict(BaseModel):
    """Claim-by-claim check of the answer against the retrieved context."""
    claims: list[Claim] = Field(description="Every factual claim in the answer (empty if it makes none)")
    reasoning: str = Field(description="One or two sentences summarising the verdict")

    @property
    def score(self) -> float:
        return sum(c.supported for c in self.claims) / len(self.claims) if self.claims else 1.0

    @property
    def unsupported(self) -> list[str]:
        return [c.claim for c in self.claims if not c.supported]


class CorrectnessVerdict(BaseModel):
    """The answer compared with the reference answer."""
    verdict: Literal["correct", "partially_correct", "incorrect"]
    missing_facts: list[str] = Field(description="Key facts of the reference the answer leaves out")
    wrong_facts: list[str] = Field(description="Statements in the answer that contradict the reference or are invented")
    reasoning: str = Field(description="One or two sentences explaining the verdict")

    @property
    def score(self) -> float:
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}[self.verdict]


faithfulness_judge = judge_llm.with_structured_output(FaithfulnessVerdict)
correctness_judge = judge_llm.with_structured_output(CorrectnessVerdict)

_cache: dict[tuple, BaseModel] = {}
_cache_lock = threading.Lock()
JUDGE_CONFIG = {"run_name": "answer-judge", "tags": ["answer-eval", "judge"]}


def _cached(key: tuple, grade):
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    verdict = grade()
    with _cache_lock:
        _cache[key] = verdict
    return verdict


def format_context(contexts: list[str]) -> str:
    if not contexts:
        return "(no knowledge-base search was made)"
    return "\n\n".join(f"--- search result {i} ---\n{c}" for i, c in enumerate(contexts, start=1))


def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> FaithfulnessVerdict:
    def grade() -> FaithfulnessVerdict:
        return faithfulness_judge.invoke([
            SystemMessage(FAITHFULNESS_PROMPT),
            HumanMessage(f"QUESTION:\n{question}\n\nCONTEXT:\n{format_context(contexts)}\n\nANSWER:\n{answer}"),
        ], config=JUDGE_CONFIG)
    return _cached(("faithfulness", question, answer, tuple(contexts)), grade)


def judge_correctness(question: str, answer: str, reference: str) -> CorrectnessVerdict:
    def grade() -> CorrectnessVerdict:
        return correctness_judge.invoke([
            SystemMessage(CORRECTNESS_PROMPT),
            HumanMessage(f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nMOXA'S ANSWER:\n{answer}"),
        ], config=JUDGE_CONFIG)
    return _cached(("correctness", question, answer, reference), grade)
