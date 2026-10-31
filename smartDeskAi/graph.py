"""
graph.py — StateGraph builder & compiler for SmartDeskAi (Moxa).

    START -> orchestrator --Send()--> hr_agent ┐
                          --Send()--> it_agent ┴-> synthesizer -> END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from smartDeskAi.agents import hr_agent, it_agent, orchestrator, synthesizer
from smartDeskAi.state import SmartDeskState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(SmartDeskState)

    builder.add_node("orchestrator", orchestrator, destinations=("hr_agent", "it_agent"))
    builder.add_node("hr_agent", hr_agent)
    builder.add_node("it_agent", it_agent)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "orchestrator")
    # orchestrator dispatches with Command(goto=[Send(...)]), so it has no static outgoing edges
    builder.add_edge("hr_agent", "synthesizer")
    builder.add_edge("it_agent", "synthesizer")
    builder.add_edge("synthesizer", END)

    # The checkpointer keeps per-session memory (thread_id) and enables interrupt()/resume
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
