"""
evaluation/retrieval_eval.py — Evaluate the hybrid RAG retriever against labeled queries.

    uv run moxa-eval-retrieval [--dataset PATH] [--min-hit-rate 0.9] [--output PATH] [--langsmith]

Each case is an employee question, the knowledge base it is searched in, and the KB entries that answer it
(any of them counts as a hit). Out-of-scope cases list no entries: the retriever should return nothing for
them, so the agent says it doesn't know instead of answering from an unrelated entry. Only the retriever
runs, against the real Qdrant collections; no LLM answers are generated.

Every stage of the pipeline is scored on the answerable cases, so each one's contribution is visible:
  dense (embeddings) · sparse (BM25) · hybrid (RRF fusion) · reranked (cross-encoder)
with Hit@1, Hit@k (k = TOP_K, the chunks an agent sees), Recall@k and MRR over the 10 candidates.

The final pipeline (re-ranked top k, then RERANK_SCORE_THRESHOLD) is scored end to end:
- answered correctly: an expected entry is among the chunks that pass the threshold
- false rejections: answerable cases where nothing passes the threshold
- correct rejections: out-of-scope cases where nothing passes
plus a threshold sweep, to show where RERANK_SCORE_THRESHOLD trades one against the other.
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

from langchain_core.runnables import RunnableLambda

from smartDeskAi.config import EMBEDDING_MODEL, RERANK_MODEL, RERANK_SCORE_THRESHOLD, SPARSE_MODEL
from smartDeskAi.logger import LOG_DIR, setup_logger
from smartDeskAi.tools.rag import (
    CANDIDATES_PER_RETRIEVER, DENSE_VECTOR_NAME, KB_FILES, SPARSE_VECTOR_NAME, TOP_K, get_knowledge_base, kb_entries,
)

logger = setup_logger("smartDeskAi.evaluation.retrieval")

DEFAULT_DATASET = Path(__file__).resolve().parent / "retrieval_dataset.json"
STAGES = ("dense", "sparse", "hybrid", "reranked")
OUT_OF_SCOPE = "out_of_scope"
THRESHOLD_SWEEP = (-10.0, -8.0, -7.0, -6.0, -5.5, -5.0, -4.5, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)


@dataclass
class RetrievalResult:
    id: str
    kb: str
    category: str
    query: str
    expected: list[str]
    returned: list[str]                    # what the agent gets: re-ranked top k that pass the threshold
    correct: bool                          # answerable: an expected id returned; out of scope: nothing returned
    rank: int | None                       # 1-based position of the first expected id after re-ranking
    top_score: float | None                # best re-rank score
    latency_s: float
    rankings: dict[str, list[str]] = field(default_factory=dict)   # every stage's candidate ids, best first
    scores: list[float] = field(default_factory=list)              # re-rank scores, aligned with rankings["reranked"]
    error: str | None = None


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    if duplicates := [i for i, n in Counter(case["id"] for case in cases).items() if n > 1]:
        raise ValueError(f"Duplicate case ids in {path}: {duplicates}")
    known = kb_entries()
    for case in cases:
        if case["kb"] not in KB_FILES:
            raise ValueError(f"Case {case['id']} searches unknown knowledge base {case['kb']!r}")
        if unknown := [kb_id for kb_id in case["expected"] if kb_id not in known]:
            raise ValueError(f"Case {case['id']} expects entries that don't exist: {unknown}")
        if wrong_kb := [kb_id for kb_id in case["expected"] if not kb_id.startswith(f"{case['kb']}_")]:
            raise ValueError(f"Case {case['id']} searches {case['kb']} but expects {wrong_kb}")
        if (case.get("category") == OUT_OF_SCOPE) != (not case["expected"]):
            raise ValueError(f"Case {case['id']}: out-of-scope cases, and only they, must expect no entries")
    return cases


# ── Running the retriever ───────────────────────────────────

def trace_case(kb: str, query: str) -> dict:
    """Every stage's ranking for one query. The local run and the LangSmith target both use this."""
    knowledge_base = get_knowledge_base(kb)
    vectors = knowledge_base.embed(query)
    fused, ranked = knowledge_base.rank(query, vectors)
    return {
        "dense": knowledge_base.single_retriever(vectors[0], DENSE_VECTOR_NAME),
        "sparse": knowledge_base.single_retriever(vectors[1], SPARSE_VECTOR_NAME),
        "hybrid": fused,
        "reranked": [chunk.id for chunk in ranked],
        "scores": [round(chunk.rerank_score, 4) for chunk in ranked],
    }


def returned_ids(trace: dict, threshold: float = RERANK_SCORE_THRESHOLD) -> list[str]:
    """What the agent would see: the re-ranked top k whose score passes the threshold (as KnowledgeBase.search)."""
    return [kb_id for kb_id, score in zip(trace["reranked"][:TOP_K], trace["scores"][:TOP_K]) if score >= threshold]


def run_retriever(cases: list[dict], concurrency: int) -> list[tuple[dict | Exception, float]]:
    """Trace every case; returns (trace or error, latency) per case."""
    for kb in {case["kb"] for case in cases}:
        get_knowledge_base(kb)  # build or open the collections up front, outside the timings

    def timed(case: dict) -> tuple[dict | Exception, float]:
        start = time.perf_counter()
        try:
            trace = trace_case(case["kb"], case["query"])
        except Exception as e:  # scored as a failed case rather than aborting the run
            logger.exception("Retrieval failed during evaluation")
            trace = e
        return trace, time.perf_counter() - start

    print(f"Retrieving {len(cases)} queries...", file=sys.stderr)
    return RunnableLambda(timed).batch(cases, config={"max_concurrency": concurrency})


def first_hit(ranking: list[str], expected: list[str]) -> int | None:
    return next((i for i, kb_id in enumerate(ranking, start=1) if kb_id in expected), None)


def score_cases(cases: list[dict], runs: list[tuple[dict | Exception, float]]) -> list[RetrievalResult]:
    results = []
    for case, (trace, latency) in zip(cases, runs):
        failed = isinstance(trace, Exception)
        trace = {stage: [] for stage in (*STAGES, "scores")} if failed else trace
        returned = returned_ids(trace)
        expected = case["expected"]
        results.append(RetrievalResult(
            id=case["id"],
            kb=case["kb"],
            category=case.get("category", "uncategorized"),
            query=case["query"],
            expected=expected,
            returned=returned,
            correct=not failed and (bool(set(returned) & set(expected)) if expected else not returned),
            rank=first_hit(trace["reranked"], expected),
            top_score=trace["scores"][0] if trace["scores"] else None,
            latency_s=round(latency, 3),
            rankings={stage: trace[stage] for stage in STAGES},
            scores=trace["scores"],
            error=f"{type(trace).__name__}: {trace}" if failed else None,
        ))
    return results


# ── Metrics ─────────────────────────────────────────────────

def ranking_metrics(pairs: list[tuple[list[str], list[str]]], k: int = TOP_K) -> dict[str, float]:
    """Hit@1, Hit@k, Recall@k and MRR over (ranking, expected ids) pairs of answerable cases."""
    if not pairs:
        return {"hit@1": 0.0, f"hit@{k}": 0.0, f"recall@{k}": 0.0, "mrr": 0.0}
    ranks = [first_hit(ranking, expected) for ranking, expected in pairs]
    return {
        "hit@1": sum(r == 1 for r in ranks) / len(pairs),
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / len(pairs),
        f"recall@{k}": statistics.mean(len(set(ranking[:k]) & set(expected)) / len(expected) for ranking, expected in pairs),
        "mrr": statistics.mean(1 / r if r else 0.0 for r in ranks),
    }


def threshold_outcomes(results: list[RetrievalResult], threshold: float) -> dict[str, float]:
    """End-to-end outcomes if RERANK_SCORE_THRESHOLD were `threshold`. Failed retrievals count as wrong."""
    def returned(r: RetrievalResult) -> list[str]:
        return returned_ids({"reranked": r.rankings["reranked"], "scores": r.scores}, threshold)

    answered = rejected = correctly_rejected = 0
    for r in results:
        if r.error:
            continue
        ids = returned(r)
        if not r.expected:
            correctly_rejected += not ids
        elif not ids:
            rejected += 1
        elif set(ids) & set(r.expected):
            answered += 1
    answerable = max(sum(bool(r.expected) for r in results), 1)
    out_of_scope = max(sum(not r.expected for r in results), 1)
    return {
        "threshold": threshold,
        "answered_correctly": answered / answerable,
        "false_rejections": rejected / answerable,
        "correct_rejections": correctly_rejected / out_of_scope,
    }


def summarize(results: list[RetrievalResult]) -> dict:
    answerable = [r for r in results if r.expected]
    out_of_scope = [r for r in results if not r.expected]
    latencies = sorted(r.latency_s for r in results)
    per_category = defaultdict(list)
    for r in results:
        per_category[r.category].append(r.correct)
    pipeline = threshold_outcomes(results, RERANK_SCORE_THRESHOLD)
    return {
        "models": {"embedding": EMBEDDING_MODEL, "sparse": SPARSE_MODEL, "reranker": RERANK_MODEL},
        "top_k": TOP_K,
        "candidates": CANDIDATES_PER_RETRIEVER,
        "threshold": RERANK_SCORE_THRESHOLD,
        "cases": len(results),
        "answerable": len(answerable),
        "out_of_scope": len(out_of_scope),
        "accuracy": sum(r.correct for r in results) / len(results),
        "answered_correctly": pipeline["answered_correctly"],
        "false_rejections": pipeline["false_rejections"],
        "correct_rejections": pipeline["correct_rejections"],
        "errors": sum(r.error is not None for r in results),
        "stages": {
            stage: ranking_metrics([(r.rankings[stage], r.expected) for r in answerable]) for stage in STAGES
        },
        "per_kb": {
            kb: ranking_metrics([(r.rankings["reranked"], r.expected) for r in answerable if r.kb == kb])
            for kb in sorted({r.kb for r in answerable})
        },
        "per_category": {
            category: {"accuracy": sum(oks) / len(oks), "cases": len(oks)} for category, oks in sorted(per_category.items())
        },
        "threshold_sweep": [threshold_outcomes(results, t) for t in sorted({*THRESHOLD_SWEEP, RERANK_SCORE_THRESHOLD})],
        "latency_s": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))],
        },
    }


def print_report(summary: dict, results: list[RetrievalResult]) -> None:
    def pct(value: float) -> str:
        return f"{value:6.1%}"

    k = summary["top_k"]
    print(f"\nRetrieval evaluation · {summary['cases']} queries ({summary['answerable']} answerable, "
          f"{summary['out_of_scope']} out of scope)")
    print(f"embeddings {summary['models']['embedding']} · reranker {summary['models']['reranker']} · "
          f"top {k} · threshold {summary['threshold']}")
    print("=" * 72)
    print(f"Pipeline accuracy      {pct(summary['accuracy'])}   (right entries returned, or nothing when out of scope)")
    print(f"Answered correctly     {pct(summary['answered_correctly'])}   of answerable queries")
    print(f"False rejections       {pct(summary['false_rejections'])}   answerable, but nothing passed the threshold")
    print(f"Correct rejections     {pct(summary['correct_rejections'])}   out of scope, and nothing returned")
    if summary["errors"]:
        print(f"Retriever errors       {summary['errors']}")
    lat = summary["latency_s"]
    print(f"Latency                mean {lat['mean']:.2f}s · p50 {lat['p50']:.2f}s · p95 {lat['p95']:.2f}s")

    print(f"\nStage (answerable)      Hit@1   Hit@{k}  Recall@{k}    MRR")
    for stage, m in summary["stages"].items():
        print(f"  {stage:<20} {pct(m['hit@1'])}  {pct(m[f'hit@{k}'])}  {pct(m[f'recall@{k}'])}  {m['mrr']:6.3f}")

    print("\nPer category           accuracy   cases")
    for category, m in summary["per_category"].items():
        print(f"  {category:<20} {pct(m['accuracy'])}   {m['cases']:>5}")

    print("\nThreshold sweep        answered  false rej.  correct rej.")
    for row in summary["threshold_sweep"]:
        marker = "  <- current" if row["threshold"] == summary["threshold"] else ""
        print(f"  {row['threshold']:>6.1f}             {pct(row['answered_correctly'])}     "
              f"{pct(row['false_rejections'])}        {pct(row['correct_rejections'])}{marker}")

    failures = [r for r in results if not r.correct]
    print(f"\nFailed queries ({len(failures)})")
    for r in failures:
        print(f"  [{r.id}] {r.query!r}")
        top = ", ".join(f"{i} ({s:.2f})" for i, s in zip(r.rankings["reranked"][:k], r.scores[:k]))
        print(f"      expected {r.expected or 'nothing'}, returned {r.returned or 'nothing'}; top {k}: {top or r.error}")
    if not failures:
        print("  none")


# ── Entry points ────────────────────────────────────────────

@dataclass
class RetrievalEvalReport:
    summary: dict
    results: list[RetrievalResult]
    output: Path                        # JSON file with the summary and every case result
    experiment_url: str | None = None   # LangSmith experiment, when run with langsmith=True


def evaluate_retrieval(
    dataset: Path = DEFAULT_DATASET,
    *,
    concurrency: int = 4,
    langsmith: bool = False,
    langsmith_dataset: str | None = None,
    output: Path | None = None,
) -> RetrievalEvalReport:
    """Run the retrieval evaluation (locally, or as a LangSmith experiment) and save the results as JSON.

    Shared by the `moxa-eval-retrieval` command and the web UI's Evaluate view.
    """
    cases = load_dataset(dataset)
    experiment_url = None
    if langsmith:
        from smartDeskAi.evaluation.retrieval_langsmith import DATASET_NAME, run_langsmith
        runs, experiment_url = run_langsmith(cases, langsmith_dataset or DATASET_NAME, concurrency)
    else:
        runs = run_retriever(cases, concurrency)
    results = score_cases(cases, runs)
    summary = summarize(results)
    if experiment_url:
        summary["langsmith_experiment"] = experiment_url

    output = output or LOG_DIR / f"retrieval_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2))
    return RetrievalEvalReport(summary, results, output, experiment_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Moxa's hybrid RAG retriever.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="labeled retrieval queries (JSON)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel queries")
    parser.add_argument("--min-hit-rate", type=float, default=None,
                        help="exit with status 1 if pipeline accuracy is below this (0-1), e.g. for CI")
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the JSON results (default: log/retrieval_eval_<timestamp>.json)")
    parser.add_argument("--langsmith", action="store_true",
                        help="run as a LangSmith experiment (syncs the queries to a LangSmith dataset first)")
    parser.add_argument("--langsmith-dataset", default=None,
                        help="LangSmith dataset name (default: 'Moxa retrieval')")
    args = parser.parse_args()

    try:
        report = evaluate_retrieval(
            args.dataset, concurrency=args.concurrency, langsmith=args.langsmith,
            langsmith_dataset=args.langsmith_dataset, output=args.output,
        )
    except ValueError as e:
        parser.error(str(e))
    summary = report.summary
    print_report(summary, report.results)
    print(f"\nFull results written to {report.output}")

    if args.min_hit_rate is not None and summary["accuracy"] < args.min_hit_rate:
        print(f"FAIL: accuracy {summary['accuracy']:.1%} is below --min-hit-rate {args.min_hit_rate:.1%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
