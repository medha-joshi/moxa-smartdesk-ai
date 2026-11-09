"""
evaluation/answer_eval.py — Evaluate the faithfulness and correctness of Moxa's answers.

    uv run moxa-eval-answers [--dataset PATH] [--min-faithfulness 0.9] [--min-correctness 0.8] [--output PATH] [--langsmith]

Each case is an employee question, a reference answer written from the knowledge base, and the KB entries
that hold it. The question runs through the whole graph, exactly as in a chat: orchestrator, HR / IT agents
with hybrid RAG, synthesizer. No employee email is set, so no Jira calls are made. The knowledge-base search
results the agents saw are captured with a callback, and the final answer is graded by an LLM judge
(JUDGE_MODEL, see answer_judges.py):

- faithfulness: share of the answer's factual claims that the retrieved context supports. An answer scoring
  below 1.0 contains at least one claim that isn't grounded in the KB (a hallucination).
- correctness: correct (1.0) / partially correct (0.5) / incorrect (0.0) against the reference answer.

Plus deterministic checks on the agents' raw citation markers:
- citation validity: share of cited entries that were actually in the search results
- source hit: answerable cases citing an expected entry (for mixed questions, one per knowledge base)
- abstention: out-of-scope cases answered without citations (correct), answerable cases without any (false)
"""

import argparse
import json
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph.state import CompiledStateGraph

from smartDeskAi.config import JUDGE_MODEL, LLM_MODEL, RERANK_SCORE_THRESHOLD
from smartDeskAi.evaluation.answer_judges import judge_correctness, judge_faithfulness
from smartDeskAi.graph import build_graph
from smartDeskAi.logger import LOG_DIR, setup_logger
from smartDeskAi.tools.hr_tools import search_hr_knowledge_base
from smartDeskAi.tools.it_tools import search_it_knowledge_base
from smartDeskAi.tools.rag import CITATION, KB_FILES, SEARCH_RESULT, TOP_K, get_knowledge_base, kb_entries

logger = setup_logger("smartDeskAi.evaluation.answers")

DEFAULT_DATASET = Path(__file__).resolve().parent / "answer_dataset.json"
OUT_OF_SCOPE = "out_of_scope"
SEARCH_TOOLS = {search_hr_knowledge_base.name, search_it_knowledge_base.name}
SOURCES_HEADING = "\n\nSources:\n"


@dataclass
class AnswerResult:
    id: str
    category: str
    question: str
    reference: str
    expected_sources: list[str]
    answer: str                             # Moxa's final reply, without the Sources list
    agents: list[str]                       # agents the orchestrator routed to
    contexts: list[str]                     # every knowledge-base search result the agents saw
    retrieved: list[str]                    # KB ids in those results
    cited: list[str]                        # KB ids the agents cited
    faithfulness: float | None              # supported claims / claims (None if the case failed)
    claims: int
    unsupported_claims: list[str]
    faithfulness_reasoning: str
    correctness: float | None               # 1.0 correct, 0.5 partially correct, 0.0 incorrect
    correctness_verdict: str
    missing_facts: list[str]
    wrong_facts: list[str]
    correctness_reasoning: str
    citation_validity: float | None         # share of cited ids that were retrieved (None if nothing cited)
    source_hit: bool | None                 # answerable only: an expected entry was cited (per KB)
    abstained: bool                         # nothing cited: "I don't have enough information" or small talk
    latency_s: float
    error: str | None = None
    agent_answers: list[dict] = field(default_factory=list)


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if duplicates := [i for i, n in Counter(case["id"] for case in cases).items() if n > 1]:
        raise ValueError(f"Duplicate case ids in {path}: {duplicates}")
    known = kb_entries()
    for case in cases:
        if not case.get("question") or not case.get("reference"):
            raise ValueError(f"Case {case['id']} needs a question and a reference answer")
        if unknown := [kb_id for kb_id in case["expected_sources"] if kb_id not in known]:
            raise ValueError(f"Case {case['id']} expects entries that don't exist: {unknown}")
        if (case.get("category") == OUT_OF_SCOPE) != (not case["expected_sources"]):
            raise ValueError(f"Case {case['id']}: out-of-scope cases, and only they, must expect no sources")
    return cases


# ── Running Moxa ────────────────────────────────────────────

class SearchRecorder(BaseCallbackHandler):
    """Records the output of every knowledge-base search made during one graph run."""

    def __init__(self) -> None:
        self.pending: dict[UUID, str] = {}
        self.results: list[str] = []

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name")
        if name in SEARCH_TOOLS:
            self.pending[run_id] = name

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if self.pending.pop(run_id, None):
            self.results.append(output.content if hasattr(output, "content") else str(output))


def strip_sources(text: str) -> str:
    """The reply without the Sources list the synthesizer appends (it only restates the citations)."""
    return text.split(SOURCES_HEADING, 1)[0].strip()


def cited_ids(texts: list[str]) -> list[str]:
    ids = [kb_id for text in texts for m in CITATION.finditer(text) for kb_id in m.group(2).replace(";", ",").split(",")]
    return list(dict.fromkeys(kb_id.strip() for kb_id in ids))


def ask_moxa(graph: CompiledStateGraph, question: str) -> dict:
    """One fresh chat session, one question. The local run and the LangSmith target both use this."""
    recorder = SearchRecorder()
    config = {
        "configurable": {"thread_id": f"answer-eval-{uuid.uuid4()}"},
        "callbacks": [recorder],
        "run_name": "answer-eval",
        "tags": ["answer-eval"],
    }
    state = graph.invoke({"messages": [HumanMessage(question)]}, config=config)
    agent_answers = [dict(r) for r in state.get("agent_responses") or []]
    if "__interrupt__" in state:  # a ticket confirmation; shouldn't happen without an email, but don't hang
        final = "(paused for ticket confirmation) " + " ".join(str(i.value) for i in state["__interrupt__"])
    else:
        final = state.get("final_response") or ""
    contexts = recorder.results
    return {
        "answer": strip_sources(final),
        "agents": state.get("selected_agents") or [],
        "agent_answers": agent_answers,
        "contexts": contexts,
        "retrieved": list(dict.fromkeys(kb_id for c in contexts for kb_id, _ in SEARCH_RESULT.findall(c))),
        "cited": cited_ids([r["response"] for r in agent_answers]),
    }


def run_moxa(cases: list[dict], concurrency: int) -> list[tuple[dict | Exception, float]]:
    """Ask Moxa every question; returns (trace or error, latency) per case."""
    for kb in KB_FILES:
        get_knowledge_base(kb)  # build or open the collections up front, outside the timings
    graph = build_graph()

    def timed(case: dict) -> tuple[dict | Exception, float]:
        start = time.perf_counter()
        try:
            trace = ask_moxa(graph, case["question"])
        except Exception as e:  # scored as a failed case rather than aborting the run
            logger.exception("Moxa failed during the answer evaluation")
            trace = e
        return trace, time.perf_counter() - start

    print(f"Asking Moxa {len(cases)} questions...", file=sys.stderr)
    return RunnableLambda(timed).batch(cases, config={"max_concurrency": concurrency})


# ── Scoring ─────────────────────────────────────────────────

def citation_validity(cited: list[str], retrieved: list[str]) -> float | None:
    return sum(kb_id in retrieved for kb_id in cited) / len(cited) if cited else None


def source_hit(cited: list[str], expected: list[str]) -> bool | None:
    """An expected entry was cited, for every knowledge base the question needs (mixed HR + IT questions)."""
    if not expected:
        return None
    needed = {kb_id.split("_", 1)[0] for kb_id in expected}
    return all(any(kb_id in cited for kb_id in expected if kb_id.startswith(f"{kb}_")) for kb in needed)


def score_case(case: dict, trace: dict) -> dict:
    """Judge one answer. Verdicts are cached, so LangSmith evaluators and the local report share them."""
    faith = judge_faithfulness(case["question"], trace["answer"], trace["contexts"])
    correct = judge_correctness(case["question"], trace["answer"], case["reference"])
    return {"faithfulness": faith, "correctness": correct}


def score_cases(cases: list[dict], runs: list[tuple[dict | Exception, float]], concurrency: int) -> list[AnswerResult]:
    def judge(pair: tuple[dict, tuple[dict | Exception, float]]) -> dict | Exception | None:
        case, (trace, _) = pair
        if isinstance(trace, Exception):
            return None
        try:
            return score_case(case, trace)
        except Exception as e:
            logger.exception("Judge failed for case %s", case["id"])
            return e

    print(f"Grading {len(cases)} answers with {JUDGE_MODEL}...", file=sys.stderr)
    verdicts = RunnableLambda(judge).batch(list(zip(cases, runs)), config={"max_concurrency": concurrency})

    results = []
    for case, (trace, latency), verdict in zip(cases, runs, verdicts):
        error = trace if isinstance(trace, Exception) else verdict if isinstance(verdict, Exception) else None
        trace = {} if isinstance(trace, Exception) else trace
        faith = correct = None
        if not error:
            faith, correct = verdict["faithfulness"], verdict["correctness"]
        cited, retrieved = trace.get("cited", []), trace.get("retrieved", [])
        results.append(AnswerResult(
            id=case["id"],
            category=case.get("category", "uncategorized"),
            question=case["question"],
            reference=case["reference"],
            expected_sources=case["expected_sources"],
            answer=trace.get("answer", ""),
            agents=trace.get("agents", []),
            contexts=trace.get("contexts", []),
            retrieved=retrieved,
            cited=cited,
            faithfulness=round(faith.score, 4) if faith else None,
            claims=len(faith.claims) if faith else 0,
            unsupported_claims=faith.unsupported if faith else [],
            faithfulness_reasoning=faith.reasoning if faith else "",
            correctness=correct.score if correct else None,
            correctness_verdict=correct.verdict if correct else "error",
            missing_facts=correct.missing_facts if correct else [],
            wrong_facts=correct.wrong_facts if correct else [],
            correctness_reasoning=correct.reasoning if correct else "",
            citation_validity=citation_validity(cited, retrieved),
            source_hit=source_hit(cited, case["expected_sources"]),
            abstained=not cited,
            latency_s=round(latency, 3),
            error=f"{type(error).__name__}: {error}" if error else None,
            agent_answers=trace.get("agent_answers", []),
        ))
    return results


# ── Metrics ─────────────────────────────────────────────────

def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(results: list[AnswerResult]) -> dict:
    graded = [r for r in results if r.error is None]
    answerable = [r for r in results if r.expected_sources]
    out_of_scope = [r for r in results if not r.expected_sources]
    cited_total = sum(len(r.cited) for r in graded)
    cited_valid = sum(sum(kb_id in r.retrieved for kb_id in r.cited) for r in graded)
    by_category: dict[str, list[AnswerResult]] = defaultdict(list)
    for r in graded:
        by_category[r.category].append(r)
    latencies = sorted(r.latency_s for r in results)
    verdicts = Counter(r.correctness_verdict for r in results)
    return {
        "models": {"llm": LLM_MODEL, "judge": JUDGE_MODEL},
        "top_k": TOP_K,
        "threshold": RERANK_SCORE_THRESHOLD,
        "cases": len(results),
        "answerable": len(answerable),
        "out_of_scope": len(out_of_scope),
        "errors": len(results) - len(graded),
        # Failed cases count as 0 so a broken run can't pass the CI gates
        "faithfulness": mean([r.faithfulness or 0.0 for r in results]),
        "fully_faithful": sum(r.faithfulness == 1.0 for r in results) / len(results),
        "hallucinated_claims": sum(len(r.unsupported_claims) for r in graded),
        "claims": sum(r.claims for r in graded),
        "correctness": mean([r.correctness or 0.0 for r in results]),
        "verdicts": {v: verdicts.get(v, 0) / len(results) for v in ("correct", "partially_correct", "incorrect", "error")},
        "citation_validity": cited_valid / cited_total if cited_total else None,
        "source_hit": sum(bool(r.source_hit) for r in answerable) / max(len(answerable), 1),
        "false_abstentions": sum(r.abstained for r in answerable if not r.error) / max(len(answerable), 1),
        "correct_abstentions": sum(r.abstained for r in out_of_scope if not r.error) / max(len(out_of_scope), 1),
        "per_category": {
            category: {
                "faithfulness": mean([r.faithfulness for r in rs]),
                "correctness": mean([r.correctness for r in rs]),
                "cases": len(rs),
            }
            for category, rs in sorted(by_category.items())
        },
        "latency_s": {
            "mean": mean(latencies),
            "p50": statistics.median(latencies),
            "p95": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
        },
    }


def print_report(summary: dict, results: list[AnswerResult]) -> None:
    def pct(value: float | None) -> str:
        return "   n/a" if value is None else f"{value:6.1%}"

    v = summary["verdicts"]
    print(f"\nAnswer evaluation · {summary['cases']} questions ({summary['answerable']} answerable, "
          f"{summary['out_of_scope']} out of scope)")
    print(f"model {summary['models']['llm']} · judge {summary['models']['judge']} · "
          f"top {summary['top_k']} · threshold {summary['threshold']}")
    print("=" * 72)
    print(f"Faithfulness           {pct(summary['faithfulness'])}   mean share of claims supported by retrieved context")
    print(f"Fully faithful         {pct(summary['fully_faithful'])}   answers with no unsupported claim "
          f"({summary['hallucinated_claims']} of {summary['claims']} claims unsupported)")
    print(f"Correctness            {pct(summary['correctness'])}   vs. reference (correct 1, partial 0.5)")
    print(f"  correct / partial / incorrect   {pct(v['correct'])} / {pct(v['partially_correct'])} / {pct(v['incorrect'])}")
    print(f"Citation validity      {pct(summary['citation_validity'])}   cited entries that were retrieved")
    print(f"Source hit             {pct(summary['source_hit'])}   answerable questions citing an expected entry")
    print(f"False abstentions      {pct(summary['false_abstentions'])}   answerable, but nothing cited")
    print(f"Correct abstentions    {pct(summary['correct_abstentions'])}   out of scope, and nothing cited")
    if summary["errors"]:
        print(f"Errors                 {summary['errors']}")
    lat = summary["latency_s"]
    print(f"Latency                mean {lat['mean']:.2f}s · p50 {lat['p50']:.2f}s · p95 {lat['p95']:.2f}s")

    print("\nPer category           faithful  correct   cases")
    for category, m in summary["per_category"].items():
        print(f"  {category:<20} {pct(m['faithfulness'])}   {pct(m['correctness'])}   {m['cases']:>5}")

    flagged = [r for r in results if r.error or r.faithfulness < 1.0 or r.correctness < 1.0]
    print(f"\nFlagged answers ({len(flagged)})")
    for r in flagged:
        print(f"  [{r.id}] {r.question!r}")
        if r.error:
            print(f"      error: {r.error}")
            continue
        print(f"      faithfulness {r.faithfulness:.2f} · {r.correctness_verdict} · cited {r.cited or 'nothing'}")
        for claim in r.unsupported_claims:
            print(f"      unsupported: {claim}")
        for fact in r.wrong_facts:
            print(f"      wrong:       {fact}")
        for fact in r.missing_facts:
            print(f"      missing:     {fact}")
    if not flagged:
        print("  none")


# ── Entry points ────────────────────────────────────────────

@dataclass
class AnswerEvalReport:
    summary: dict
    results: list[AnswerResult]
    output: Path                        # JSON file with the summary and every case result
    experiment_url: str | None = None   # LangSmith experiment, when run with langsmith=True


def evaluate_answers(
    dataset: Path = DEFAULT_DATASET,
    *,
    concurrency: int = 4,
    langsmith: bool = False,
    langsmith_dataset: str | None = None,
    output: Path | None = None,
) -> AnswerEvalReport:
    """Run the answer evaluation (locally, or as a LangSmith experiment) and save the results as JSON.

    Shared by the `moxa-eval-answers` command and the web UI's Evaluate view.
    """
    cases = load_dataset(dataset)
    experiment_url = None
    if langsmith:
        from smartDeskAi.evaluation.answer_langsmith import DATASET_NAME, run_langsmith
        runs, experiment_url = run_langsmith(cases, langsmith_dataset or DATASET_NAME, concurrency)
    else:
        runs = run_moxa(cases, concurrency)
    results = score_cases(cases, runs, concurrency)  # cached verdicts when LangSmith already graded them
    summary = summarize(results)
    if experiment_url:
        summary["langsmith_experiment"] = experiment_url

    output = output or LOG_DIR / f"answer_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2))
    return AnswerEvalReport(summary, results, output, experiment_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the faithfulness and correctness of Moxa's answers.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="questions with reference answers (JSON)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel questions")
    parser.add_argument("--min-faithfulness", type=float, default=None,
                        help="exit with status 1 if mean faithfulness is below this (0-1), e.g. for CI")
    parser.add_argument("--min-correctness", type=float, default=None,
                        help="exit with status 1 if mean correctness is below this (0-1), e.g. for CI")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the JSON results (default: log/answer_eval_<timestamp>.json)")
    parser.add_argument("--langsmith", action="store_true",
                        help="run as a LangSmith experiment (syncs the questions to a LangSmith dataset first)")
    parser.add_argument("--langsmith-dataset", default=None,
                        help="LangSmith dataset name (default: 'Moxa answers')")
    args = parser.parse_args()

    try:
        report = evaluate_answers(
            args.dataset, concurrency=args.concurrency, langsmith=args.langsmith,
            langsmith_dataset=args.langsmith_dataset, output=args.output,
        )
    except ValueError as e:
        parser.error(str(e))
    summary = report.summary
    print_report(summary, report.results)
    print(f"\nFull results written to {report.output}")

    failed = False
    for name, minimum in (("faithfulness", args.min_faithfulness), ("correctness", args.min_correctness)):
        if minimum is not None and summary[name] < minimum:
            print(f"FAIL: {name} {summary[name]:.1%} is below --min-{name} {minimum:.1%}")
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
