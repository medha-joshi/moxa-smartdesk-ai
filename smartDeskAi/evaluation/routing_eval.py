"""
evaluation/routing_eval.py — Evaluate the orchestrator's routing decisions against a labeled dataset.

    uv run moxa-eval-routing [--dataset PATH] [--repeats N] [--min-accuracy 0.9] [--output PATH] [--langsmith]

Each case is a conversation (ending with the employee's latest message) and the set of agents that should
handle it. The orchestrator's own LLM router is called with the same prompt and history window it uses in
the graph, so no sub-agents, RAG or Jira calls run. The keyword fallback is scored too, as a baseline.

Metrics:
- exact-match accuracy: the selected agent set equals the expected set
- per-agent precision / recall / F1: was each agent selected when (and only when) it should be
- per-category accuracy, and a confusion table of expected vs predicted agent sets
- consistency (with --repeats > 1): share of cases where every run returned the same agent set

With --langsmith the cases are synced to a LangSmith dataset and run as a scored LangSmith experiment
(see routing_langsmith.py); the same local report is printed afterwards.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from smartDeskAi.agents.orchestrator import RoutingDecision, fallback_route, router, routing_input
from smartDeskAi.config import LLM_MODEL
from smartDeskAi.logger import LOG_DIR, setup_logger

logger = setup_logger("smartDeskAi.evaluation.routing")

DEFAULT_DATASET = Path(__file__).resolve().parent / "routing_dataset.json"
AGENTS = ("hr_agent", "it_agent")
ERROR = "error"


@dataclass
class CaseResult:
    id: str
    category: str
    latest_message: str
    expected: list[str]
    predicted: list[str] | str             # agent list, or "error" if the router call failed
    correct: bool
    reasoning: str
    latency_s: float
    runs: list[list[str] | str] = field(default_factory=list)   # every repeat's prediction
    fallback_predicted: list[str] = field(default_factory=list)
    fallback_correct: bool = False


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    ids = Counter(case["id"] for case in cases)
    if duplicates := [case_id for case_id, count in ids.items() if count > 1]:
        raise ValueError(f"Duplicate case ids in {path}: {duplicates}")
    for case in cases:
        if not case["messages"] or case["messages"][-1]["role"] != "user":
            raise ValueError(f"Case {case['id']} must end with a user message")
        if unknown := set(case["expected"]) - set(AGENTS):
            raise ValueError(f"Case {case['id']} has unknown agents: {unknown}")
    return cases


def to_messages(raw: list[dict]) -> list[AnyMessage]:
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in raw]


def normalize(agents: list[str]) -> list[str]:
    """Order-free, de-duplicated agent set, with the orchestrator's empty-list default applied."""
    return sorted(set(agents)) or ["hr_agent"]


def run_router(cases: list[dict], repeats: int, concurrency: int) -> list[list[tuple[RoutingDecision | Exception, float]]]:
    """Call the LLM router for every case `repeats` times; returns (decision or error, latency) per run."""
    inputs = [routing_input(to_messages(case["messages"])) for case in cases]

    def timed(messages: list[AnyMessage]) -> tuple[RoutingDecision | Exception, float]:
        start = time.perf_counter()
        try:
            decision = router.invoke(messages, config={"run_name": "routing-eval", "tags": ["routing-eval"]})
        except Exception as e:  # scored as a failed case rather than aborting the run
            logger.exception("Router call failed during evaluation")
            decision = e
        return decision, time.perf_counter() - start

    runner = RunnableLambda(timed)
    results: list[list] = [[] for _ in cases]
    for repeat in range(repeats):
        print(f"Routing {len(cases)} cases (run {repeat + 1}/{repeats})...", file=sys.stderr)
        for i, outcome in enumerate(runner.batch(inputs, config={"max_concurrency": concurrency})):
            results[i].append(outcome)
    return results


def score_cases(cases: list[dict], runs: list[list[tuple[RoutingDecision | Exception, float]]]) -> list[CaseResult]:
    results = []
    for case, case_runs in zip(cases, runs):
        expected = normalize(case["expected"])
        predictions = [ERROR if isinstance(d, Exception) else normalize(d.agents) for d, _ in case_runs]
        first, latency = case_runs[0]
        latest = case["messages"][-1]["content"]
        fallback = normalize(fallback_route(latest).agents)
        results.append(CaseResult(
            id=case["id"],
            category=case.get("category", "uncategorized"),
            latest_message=latest,
            expected=expected,
            predicted=predictions[0],
            correct=predictions[0] == expected,
            reasoning=f"{type(first).__name__}: {first}" if isinstance(first, Exception) else first.reasoning,
            latency_s=round(latency, 3),
            runs=predictions,
            fallback_predicted=fallback,
            fallback_correct=fallback == expected,
        ))
    return results


def per_agent_metrics(pairs: list[tuple[list[str], list[str] | str]]) -> dict[str, dict[str, float]]:
    """Precision / recall / F1 per agent over (expected, predicted) pairs; errors count as selecting nothing."""
    metrics = {}
    for agent in AGENTS:
        tp = fp = fn = 0
        for expected, predicted in pairs:
            selected = predicted != ERROR and agent in predicted
            should = agent in expected
            tp += selected and should
            fp += selected and not should
            fn += should and not selected
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[agent] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return metrics


def summarize(results: list[CaseResult]) -> dict:
    total = len(results)
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    latencies = sorted(r.latency_s for r in results)
    label = lambda agents: ERROR if agents == ERROR else "+".join(agents)  # noqa: E731
    summary = {
        "model": LLM_MODEL,
        "cases": total,
        "accuracy": sum(r.correct for r in results) / total,
        "errors": sum(r.predicted == ERROR for r in results),
        "per_agent": per_agent_metrics([(r.expected, r.predicted) for r in results]),
        "per_category": {
            category: {"accuracy": sum(r.correct for r in rs) / len(rs), "cases": len(rs)}
            for category, rs in sorted(by_category.items())
        },
        "confusion": dict(Counter(f"{label(r.expected)} -> {label(r.predicted)}" for r in results)),
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(total - 1, round(0.95 * (total - 1)))],
        },
        "keyword_fallback": {
            "accuracy": sum(r.fallback_correct for r in results) / total,
            "per_agent": per_agent_metrics([(r.expected, r.fallback_predicted) for r in results]),
        },
    }
    if len(results[0].runs) > 1:
        summary["consistency"] = sum(all(p == r.runs[0] for p in r.runs) for r in results) / total
        summary["accuracy_all_runs"] = (
            sum(p == r.expected for r in results for p in r.runs) / sum(len(r.runs) for r in results)
        )
    return summary


def print_report(summary: dict, results: list[CaseResult]) -> None:
    pct = lambda x: f"{x:6.1%}"  # noqa: E731
    print(f"\nRouting evaluation · model {summary['model']} · {summary['cases']} cases")
    print("=" * 64)
    print(f"Exact-match accuracy   {pct(summary['accuracy'])}   (keyword fallback {pct(summary['keyword_fallback']['accuracy'])})")
    if "consistency" in summary:
        print(f"Accuracy over all runs {pct(summary['accuracy_all_runs'])}")
        print(f"Consistency            {pct(summary['consistency'])}   (same agents in every run)")
    if summary["errors"]:
        print(f"Router errors          {summary['errors']}")
    lat = summary["latency_s"]
    print(f"Latency                mean {lat['mean']:.2f}s · p50 {lat['p50']:.2f}s · p95 {lat['p95']:.2f}s")

    print("\nPer agent              precision  recall     F1   support")
    for agent, m in summary["per_agent"].items():
        print(f"  {agent:<20} {pct(m['precision'])}   {pct(m['recall'])} {pct(m['f1'])}   {m['support']:>4}")

    print("\nPer category           accuracy   cases")
    for category, m in summary["per_category"].items():
        print(f"  {category:<20} {pct(m['accuracy'])}   {m['cases']:>5}")

    print("\nExpected -> predicted")
    for pair, count in sorted(summary["confusion"].items(), key=lambda kv: -kv[1]):
        print(f"  {pair:<40} {count:>3}")

    failures = [r for r in results if not r.correct]
    print(f"\nMisrouted cases ({len(failures)})")
    for r in failures:
        print(f"  [{r.id}] {r.latest_message[:70]!r}")
        print(f"      expected {r.expected}, got {r.predicted}: {r.reasoning}")
    if not failures:
        print("  none")


@dataclass
class RoutingEvalReport:
    summary: dict
    results: list[CaseResult]
    output: Path                        # JSON file with the summary and every case result
    experiment_url: str | None = None   # LangSmith experiment, when run with langsmith=True


def evaluate_routing(
    dataset: Path = DEFAULT_DATASET,
    *,
    repeats: int = 1,
    concurrency: int = 8,
    langsmith: bool = False,
    langsmith_dataset: str | None = None,
    output: Path | None = None,
) -> RoutingEvalReport:
    """Run the routing evaluation (locally, or as a LangSmith experiment) and save the results as JSON.

    Shared by the `moxa-eval-routing` command and the web UI's Evaluate view.
    """
    cases = load_dataset(dataset)
    experiment_url = None
    if langsmith:
        from smartDeskAi.evaluation.routing_langsmith import DATASET_NAME, run_langsmith
        runs, experiment_url = run_langsmith(cases, langsmith_dataset or DATASET_NAME, repeats, concurrency)
    else:
        runs = run_router(cases, repeats, concurrency)
    results = score_cases(cases, runs)
    summary = summarize(results)
    if experiment_url:
        summary["langsmith_experiment"] = experiment_url

    output = output or LOG_DIR / f"routing_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2))
    return RoutingEvalReport(summary, results, output, experiment_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Moxa's orchestrator routing decisions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="labeled routing cases (JSON)")
    parser.add_argument("--repeats", type=int, default=1, help="run each case N times to measure consistency")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel router calls")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="exit with status 1 if exact-match accuracy is below this (0-1), e.g. for CI")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the JSON results (default: log/routing_eval_<timestamp>.json)")
    parser.add_argument("--langsmith", action="store_true",
                        help="run as a LangSmith experiment (syncs the cases to a LangSmith dataset first)")
    parser.add_argument("--langsmith-dataset", default=None,
                        help="LangSmith dataset name (default: 'Moxa routing decisions')")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    try:
        report = evaluate_routing(
            args.dataset, repeats=args.repeats, concurrency=args.concurrency,
            langsmith=args.langsmith, langsmith_dataset=args.langsmith_dataset, output=args.output,
        )
    except ValueError as e:
        parser.error(str(e))
    summary = report.summary
    print_report(summary, report.results)
    print(f"\nFull results written to {report.output}")

    if args.min_accuracy is not None and summary["accuracy"] < args.min_accuracy:
        print(f"FAIL: accuracy {summary['accuracy']:.1%} is below --min-accuracy {args.min_accuracy:.1%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
