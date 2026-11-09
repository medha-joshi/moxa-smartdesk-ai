"""
evaluation/answer_langsmith.py — Run the answer evaluation as a LangSmith experiment.

Used by `uv run moxa-eval-answers --langsmith`. The local JSON questions are synced to a LangSmith dataset
(matched by case id, as for routing), then Moxa's full graph is run over it with `Client.evaluate`. Each
run's outputs hold the answer, the routed agents, the retrieved context and the citations, and it is scored:

- per example: faithfulness and correctness (LLM judge, with the judge's reasoning as the comment),
  citation_validity, source_hit (answerable questions), abstained_correctly
- per experiment: mean faithfulness, mean correctness, fully_faithful, source_hit, false / correct abstentions

Experiments carry the answering and judge models, top k and threshold as metadata. The judge verdicts are
cached, so the local report printed afterwards reuses them instead of grading every answer again.
"""

import hashlib
from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph
from langsmith import Client
from langsmith.schemas import Example, Run

from smartDeskAi.agents.prompts import SYNTHESIZER_PROMPT, _SPECIALIST_PROMPT
from smartDeskAi.config import JUDGE_MODEL, LLM_MODEL, RERANK_SCORE_THRESHOLD
from smartDeskAi.evaluation.answer_eval import OUT_OF_SCOPE, ask_moxa, citation_validity, source_hit
from smartDeskAi.evaluation.answer_judges import judge_correctness, judge_faithfulness
from smartDeskAi.evaluation.routing_langsmith import example_id, experiment_url, langsmith_configured, sync_dataset
from smartDeskAi.graph import build_graph
from smartDeskAi.tools.rag import KB_FILES, TOP_K, get_knowledge_base

DATASET_NAME = "Moxa answers"


def to_example(dataset_name: str, case: dict) -> dict:
    return {
        "id": example_id(dataset_name, case["id"]),
        "inputs": {"question": case["question"]},
        "outputs": {"reference": case["reference"], "expected_sources": case["expected_sources"]},
        "metadata": {"case_id": case["id"], "category": case.get("category", "uncategorized")},
    }


# ── Target and evaluators ───────────────────────────────────

@lru_cache(maxsize=1)
def graph() -> CompiledStateGraph:
    for kb in KB_FILES:
        get_knowledge_base(kb)
    return build_graph()


def answer(inputs: dict) -> dict:
    return ask_moxa(graph(), inputs["question"])


def faithfulness(inputs: dict, outputs: dict | None) -> dict:
    if not outputs:
        return {"key": "faithfulness", "score": 0.0, "comment": "no answer"}
    verdict = judge_faithfulness(inputs["question"], outputs["answer"], outputs["contexts"])
    unsupported = "".join(f"\n- {claim}" for claim in verdict.unsupported)
    return {"key": "faithfulness", "score": verdict.score,
            "comment": verdict.reasoning + (f"\nUnsupported:{unsupported}" if unsupported else "")}


def correctness(inputs: dict, outputs: dict | None, reference_outputs: dict) -> dict:
    if not outputs:
        return {"key": "correctness", "score": 0.0, "comment": "no answer"}
    verdict = judge_correctness(inputs["question"], outputs["answer"], reference_outputs["reference"])
    details = "".join(f"\nWrong: {f}" for f in verdict.wrong_facts) + "".join(f"\nMissing: {f}" for f in verdict.missing_facts)
    return {"key": "correctness", "score": verdict.score, "value": verdict.verdict, "comment": verdict.reasoning + details}


def citations_valid(outputs: dict | None) -> dict:
    score = citation_validity((outputs or {}).get("cited") or [], (outputs or {}).get("retrieved") or [])
    return {"key": "citation_validity", "score": score, **({"comment": "nothing cited"} if score is None else {})}


def cited_expected_source(outputs: dict | None, reference_outputs: dict) -> dict:
    hit = source_hit((outputs or {}).get("cited") or [], reference_outputs["expected_sources"])
    return {"key": "source_hit", "score": hit, **({"comment": "out of scope"} if hit is None else {})}


def abstained_correctly(outputs: dict | None, reference_outputs: dict) -> dict:
    abstained = not (outputs or {}).get("cited")
    return {"key": "abstained_correctly", "score": outputs is not None and abstained == (not reference_outputs["expected_sources"])}


def answer_summary(inputs: list[dict], outputs: list[dict | None], reference_outputs: list[dict]) -> dict:
    rows = list(zip(inputs, outputs, reference_outputs))
    faith = [faithfulness(i, o)["score"] for i, o, _ in rows]
    correct = [correctness(i, o, r)["score"] for i, o, r in rows]
    answerable = [(o, r) for _, o, r in rows if r["expected_sources"]]
    out_of_scope = [o for _, o, r in rows if not r["expected_sources"]]
    cited = lambda o: (o or {}).get("cited") or []  # noqa: E731
    return {"results": [
        {"key": "mean_faithfulness", "score": sum(faith) / len(faith)},
        {"key": "fully_faithful", "score": sum(s == 1.0 for s in faith) / len(faith)},
        {"key": "mean_correctness", "score": sum(correct) / len(correct)},
        {"key": "source_hit", "score": sum(bool(source_hit(cited(o), r["expected_sources"])) for o, r in answerable) / max(len(answerable), 1)},
        {"key": "false_abstentions", "score": sum(o is not None and not cited(o) for o, _ in answerable) / max(len(answerable), 1)},
        {"key": "correct_abstentions", "score": sum(o is not None and not cited(o) for o in out_of_scope) / max(len(out_of_scope), 1)},
    ]}


def run_langsmith(cases: list[dict], dataset_name: str, concurrency: int) -> tuple[list[tuple[dict | Exception, float]], str | None]:
    """Run the experiment in LangSmith.

    Returns per-case (trace or error, latency) for the local report, and the experiment's URL.
    """
    if not langsmith_configured():
        raise ValueError("LangSmith evaluation needs LANGSMITH_API_KEY in your .env file or environment.")
    client = Client()
    dataset_id = sync_dataset(
        client, dataset_name, [to_example(dataset_name, case) for case in cases],
        "Employee questions with reference answers from the Moxa knowledge base "
        f"({OUT_OF_SCOPE} questions should get an honest 'I don't have enough information').",
    )
    prompt_hash = hashlib.sha256((_SPECIALIST_PROMPT + SYNTHESIZER_PROMPT).encode()).hexdigest()[:8]

    results = client.evaluate(
        answer,
        data=dataset_name,
        evaluators=[faithfulness, correctness, citations_valid, cited_expected_source, abstained_correctly],
        summary_evaluators=[answer_summary],
        experiment_prefix=f"answers-{LLM_MODEL}",
        description="Answer faithfulness & correctness (moxa-eval-answers --langsmith)",
        metadata={
            "model": LLM_MODEL, "judge_model": JUDGE_MODEL, "prompt_sha": prompt_hash,
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
