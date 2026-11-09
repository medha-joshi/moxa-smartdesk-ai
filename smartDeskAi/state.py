"""
state.py — Shared graph state for SmartDeskAi.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentResponse(TypedDict):
    agent: str      # "hr_agent" or "it_agent"
    response: str


def merge_agent_responses(left: list[AgentResponse] | None, right: list[AgentResponse] | None) -> list[AgentResponse]:
    """Collect responses from agents running in parallel; a None update clears them for a new turn."""
    if right is None:
        return []
    return (left or []) + right


class SmartDeskState(TypedDict, total=False):
    # Full conversation for this session (employee messages + Moxa's final replies)
    messages: Annotated[list[AnyMessage], add_messages]
    # Remembered for the whole session so Moxa never re-asks
    employee_email: str | None
    # True when employee_email comes from sign-in; chat messages can't change it then
    email_verified: bool
    # Orchestrator routing decision for the current turn
    selected_agents: list[str]
    routing_reasoning: str
    # Answers produced in parallel by the sub-agents for the current turn
    agent_responses: Annotated[list[AgentResponse], merge_agent_responses]
    final_response: str


class AgentTask(TypedDict):
    """Payload the orchestrator sends to each sub-agent with Send()."""
    messages: list[AnyMessage]
    employee_email: str | None
    question: str | None    # this agent's part of a mixed HR + IT message, written by the router
