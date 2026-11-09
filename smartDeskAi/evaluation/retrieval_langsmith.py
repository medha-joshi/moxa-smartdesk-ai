"""
evaluation/retrieval_langsmith.py — Run the retrieval evaluation as a LangSmith experiment.

Used by `uv run moxa-eval-retrieval --langsmith`. The local JSON queries are synced to a LangSmith dataset
(matched by case id, as for routing), then the retriever is run over it with `Client.evaluate`. Each run's
outputs hold every stage's ranking, and it is scored in LangSmith:

- per example: correct (right entries returned, or nothing when out of scope), hit_at_1, hit_at_k,
  reciprocal_rank (answerable queries only)
- per experiment: accuracy, answered_correctly, false_rejections, correct_rejections, and hit@k / MRR per stage

Experiments carry the models, top k and threshold as metadata, so a re-ranker or threshold change can be
compared side by side with the previous run in the LangSmith UI.
"""

from langsmith import Client
from langsmith.schemas import Example, Run

from smartDeskAi.config import EMBEDDING_MODEL, RERANK_MODEL, RERANK_SCORE_THRESHOLD, SPARSE_MODEL
from smartDeskAi.evaluation.retrieval_eval import (
    OUT_OF_SCOPE, STAGES, first_hit, ranking_metrics, returned_ids, trace_case,
)
from smartDeskAi.evaluation.routing_langsmith import example_id, experiment_url, langsmith_configured, sync_dataset
from smartDeskAi.tools.rag import TOP_K

DATASET_NAME = "Moxa retrieval"


def to_example(dataset_name: str, case: dict) -> dict:
    return {
        "id": example_id(dataset_name, case["id"]),
        "inputs": {"kb": case["kb"], "query": case["query"]},
        "outputs": {"expected": case["expected"]},
        "metadata": {"case_id": case["id"], "category": case.get("category", "uncategorized"), "kb": case["kb"]},
    }


# ── Target and evaluators ───────────────────────────────────

def retrieve(inputs: dict) -> dict:
    trace = trace_case(inputs["kb"], inputs["query"])
    return {**trace, "returned": returned_ids(trace)}


def reranked(outputs: dict | None) -> list[str]:
    # A failed retrieval has no outputs; score it as returning nothing
    return (outputs or {}).get("reranked") or []


def correct(outputs: dict | None, reference_outputs: dict) -> dict:
    returned, expected = (outputs or {}).get("returned") or [], reference_outputs["expected"]
    ok = outputs is not None and (bool(set(returned) & set(expected)) if expected else not returned)
    return {"key": "correct", "score": ok}


def hit_at_1(outputs: dict | None, reference_outputs: dict) -> dict:
    if not reference_outputs["expected"]:
        return {"key": "hit_at_1", "score": None, "comment": "out of scope"}
    return {"key": "hit_at_1", "score": first_hit(reranked(outputs), reference_outputs["expected"]) == 1}


def hit_at_k(outputs: dict | None, reference_outputs: dict) -> dict:
    if not reference_outputs["expected"]:
        return {"key": f"hit_at_{TOP_K}", "score": None, "comment": "out of scope"}
    rank = first_hit(reranked(outputs), reference_outputs["expected"])
    return {"key": f"hit_at_{TOP_K}", "score": rank is not None and rank <= TOP_K}


def reciprocal_rank(outputs: dict | None, reference_outputs: dict) -> dict:
    if not reference_outputs["expected"]:
        return {"key": "reciprocal_rank", "score": None, "comment": "out of scope"}
    rank = first_hit(reranked(outputs), reference_outputs["expected"])
    return {"key": "reciprocal_rank", "score": 1 / rank if rank else 0.0}


def retrieval_summary(outputs: list[dict | None], reference_outputs: list[dict]) -> dict:
    pairs = list(zip(outputs, reference_outputs))
    answerable = [(out, ref) for out, ref in pairs if ref["expected"]]
    out_of_scope = [(out, ref) for out, ref in pairs if not ref["expected"]]
    returned = lambda out: (out or {}).get("returned") or []  # noqa: E731
    results = [
        {"key": "accuracy", "score": sum(correct(out, ref)["score"] for out, ref in pairs) / len(pairs)},
        {"key": "answered_correctly",
         "score": sum(bool(set(returned(out)) & set(ref["expected"])) for out, ref in answerable) / max(len(answerable), 1)},
        {"key": "false_rejections",
         "score": sum(out is not None and not returned(out) for out, _ in answerable) / max(len(answerable), 1)},
        {"key": "correct_rejections",
         "score": sum(out is not None and not returned(out) for out, _ in out_of_scope) / max(len(out_of_scope), 1)},
    ]
    for stage in STAGES:
        m = ranking_metrics([((out or {}).get(stage) or [], ref["expected"]) for out, ref in answerable])
        results += [{"key": f"{stage}_hit_at_{TOP_K}", "score": m[f"hit@{TOP_K}"]},
                    {"key": f"{stage}_mrr", "score": m["mrr"]}]
    return {"results": results}


def run_langsmith(cases: list[dict], dataset_name: str, concurrency: int) -> tuple[list[tuple[dict | Exception, float]], str | None]:
    """Run the experiment in LangSmith.

    Returns per-case (trace or error, latency) for the local report, and the experiment's URL.
    """
    if not langsmith_configured():
        raise ValueError("LangSmith evaluation needs LANGSMITH_API_KEY in your .env file or environment.")
    client = Client()
    dataset_id = sync_dataset(
        client, dataset_name, [to_example(dataset_name, case) for case in cases],
        "Employee questions labeled with the Moxa knowledge-base entries that answer them "
        f"({OUT_OF_SCOPE} questions expect none).",
    )

    results = client.evaluate(
        retrieve,
        data=dataset_name,
        evaluators=[correct, hit_at_1, hit_at_k, reciprocal_rank],
        summary_evaluators=[retrieval_summary],
        experiment_prefix=f"retrieval-{RERANK_MODEL.rsplit('/', 1)[-1]}",
        description="Hybrid RAG retrieval (moxa-eval-retrieval --langsmith)",
        metadata={
            "embedding_model": EMBEDDING_MODEL, "sparse_model": SPARSE_MODEL, "rerank_model": RERANK_MODEL,
            "top_k": TOP_K, "threshold": RERANK_SCORE_THRESHOLD,
        },
        max_concurrency=concurrency,
    )

    # Map LangSmith rows back to the local cases, in dataset-file order
    index = {example_id(dataset_name, case["id"]): i for i, case in enumerate(cases)}
    runs: list[tuple[dict | Exception, float]] = [(RuntimeError("no LangSmith run"), 0.0)] * len(cases)
    for row in results:
        run: Run = row["run"]
        example: Example = row["example"]
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0.0
        runs[index[example.id]] = (RuntimeError(run.error) if run.error else run.outputs, latency)
    url = experiment_url(client, results.experiment_name, dataset_id)
    print(f"\nLangSmith experiment: {results.experiment_name}" + (f"\n{url}" if url else ""))
    return runs, url
