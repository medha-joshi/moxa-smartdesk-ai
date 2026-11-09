"""
evaluation/routing_langsmith.py — Run the routing evaluation as a LangSmith experiment.

Used by `uv run moxa-eval-routing --langsmith`. The local JSON dataset is synced to a LangSmith dataset
(examples are matched by case id, so edits to the file update the examples in place), then the router is run
over it with `Client.evaluate`. Each run is scored in LangSmith:

- per example: exact_match, hr_agent_correct, it_agent_correct
- per experiment: accuracy, and precision / recall / F1 for each agent

Experiments carry the model and a hash of the routing prompt as metadata, so runs before and after a
prompt or model change can be compared side by side in the LangSmith UI.
"""

import hashlib
import os
import uuid

from langsmith import Client
from langsmith.schemas import Example, Run

from smartDeskAi.agents.orchestrator import ROUTING_HISTORY, RoutingDecision, router, routing_input
from smartDeskAi.agents.prompts import ORCHESTRATOR_PROMPT
from smartDeskAi.config import LLM_MODEL
from smartDeskAi.evaluation.routing_eval import AGENTS, normalize, per_agent_metrics, to_messages
from smartDeskAi.logger import setup_logger

logger = setup_logger("smartDeskAi.evaluation.routing_langsmith")

DATASET_NAME = "Moxa routing decisions"
# Stable example ids: the same case id always maps to the same LangSmith example
EXAMPLE_NAMESPACE = uuid.UUID("6f1d3c52-8a4e-4c1b-9d7a-2b5e0f9c4a11")


def example_id(dataset_name: str, case_id: str) -> uuid.UUID:
    return uuid.uuid5(EXAMPLE_NAMESPACE, f"{dataset_name}:{case_id}")


def to_example(dataset_name: str, case: dict) -> dict:
    return {
        "id": example_id(dataset_name, case["id"]),
        "inputs": {"messages": case["messages"]},
        "outputs": {"agents": normalize(case["expected"])},
        "metadata": {"case_id": case["id"], "category": case.get("category", "uncategorized")},
    }


def sync_dataset(client: Client, dataset_name: str, wanted: list[dict], description: str) -> str:
    """Create or update the LangSmith dataset to hold exactly `wanted` examples; returns the dataset id.

    Examples need stable ids (see example_id), so edits to a case update its example in place.
    """
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(dataset_name, description=description)
    existing = {ex.id: ex for ex in client.list_examples(dataset_id=dataset.id)}

    new = [ex for ex in wanted if ex["id"] not in existing]
    changed = [
        ex for ex in wanted if ex["id"] in existing and (
            existing[ex["id"]].inputs != ex["inputs"]
            or existing[ex["id"]].outputs != ex["outputs"]
            or {k: existing[ex["id"]].metadata.get(k) for k in ex["metadata"]} != ex["metadata"]
        )
    ]
    stale = [ex_id for ex_id in existing if ex_id not in {ex["id"] for ex in wanted}]
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    if changed:
        client.update_examples(dataset_id=dataset.id, updates=changed)
    if stale:
        client.delete_examples(stale)
    print(f"LangSmith dataset {dataset_name!r}: {len(new)} added, {len(changed)} updated, "
          f"{len(stale)} removed, {len(wanted) - len(new) - len(changed)} unchanged")
    return str(dataset.id)


# ── Target and evaluators ───────────────────────────────────

def route(inputs: dict) -> dict:
    decision = router.invoke(routing_input(to_messages(inputs["messages"])))
    return {"agents": normalize(decision.agents), "reasoning": decision.reasoning}


def predicted(outputs: dict | None) -> list[str]:
    # A failed router call has no outputs; score it as selecting nothing
    return (outputs or {}).get("agents") or []


def exact_match(outputs: dict | None, reference_outputs: dict) -> dict:
    return {"key": "exact_match", "score": predicted(outputs) == reference_outputs["agents"]}


def agent_correct(agent: str):
    def evaluator(outputs: dict | None, reference_outputs: dict) -> dict:
        selected = agent in predicted(outputs)
        return {"key": f"{agent}_correct", "score": selected == (agent in reference_outputs["agents"])}
    evaluator.__name__ = f"{agent}_correct"
    return evaluator


def routing_summary(outputs: list[dict | None], reference_outputs: list[dict]) -> dict:
    pairs = [(ref["agents"], predicted(out)) for out, ref in zip(outputs, reference_outputs)]
    results = [{"key": "accuracy", "score": sum(exp == pred for exp, pred in pairs) / len(pairs)}]
    for agent, m in per_agent_metrics(pairs).items():
        results += [{"key": f"{agent}_{name}", "score": m[name]} for name in ("precision", "recall", "f1")]
    return {"results": results}


def langsmith_configured() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def run_langsmith(
    cases: list[dict], dataset_name: str, repeats: int, concurrency: int,
) -> tuple[list[list[tuple[RoutingDecision | Exception, float]]], str | None]:
    """Run the experiment in LangSmith.

    Returns per-case (decision or error, latency) runs for the local report, and the experiment's URL.
    """
    if not langsmith_configured():
        raise ValueError("LangSmith evaluation needs LANGSMITH_API_KEY in your .env file or environment.")
    client = Client()
    dataset_id = sync_dataset(
        client, dataset_name, [to_example(dataset_name, case) for case in cases],
        "Conversations labeled with the Moxa agents that should handle them.",
    )
    prompt_hash = hashlib.sha256(ORCHESTRATOR_PROMPT.encode()).hexdigest()[:8]

    results = client.evaluate(
        route,
        data=dataset_name,
        evaluators=[exact_match, *(agent_correct(agent) for agent in AGENTS)],
        summary_evaluators=[routing_summary],
        experiment_prefix=f"routing-{LLM_MODEL}",
        description="Orchestrator routing decisions (moxa-eval-routing --langsmith)",
        metadata={"model": LLM_MODEL, "prompt_sha": prompt_hash, "routing_history": ROUTING_HISTORY},
        num_repetitions=repeats,
        max_concurrency=concurrency,
    )

    # Map LangSmith rows back to the local cases, in dataset-file order
    index = {example_id(dataset_name, case["id"]): i for i, case in enumerate(cases)}
    runs: list[list] = [[] for _ in cases]
    for row in results:
        run: Run = row["run"]
        example: Example = row["example"]
        latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0.0
        outcome = RuntimeError(run.error) if run.error else RoutingDecision(**run.outputs)
        runs[index[example.id]].append((outcome, latency))
    url = experiment_url(client, results.experiment_name, dataset_id)
    print(f"\nLangSmith experiment: {results.experiment_name}" + (f"\n{url}" if url else ""))
    return runs, url


def experiment_url(client: Client, experiment_name: str, dataset_id: str) -> str | None:
    """Link to the experiment's results page (the same one `evaluate` prints)."""
    try:
        project = client.read_project(project_name=experiment_name)
    except Exception:
        logger.exception("Could not look up LangSmith experiment %r", experiment_name)
        return None
    if not project.url:
        return None
    base_url = project.url.split("?")[0].split("/projects/p/")[0]
    return f"{base_url}/datasets/{dataset_id}/compare?selectedSessions={project.id}"
