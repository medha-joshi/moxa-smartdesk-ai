"""
agents/orchestrator.py — Routes each employee message to hr_agent and/or it_agent in parallel.
"""

import re
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from smartDeskAi.agents.prompts import ORCHESTRATOR_PROMPT
from smartDeskAi.config import llm
from smartDeskAi.logger import setup_logger
from smartDeskAi.state import AgentTask, SmartDeskState

logger = setup_logger("smartDeskAi.agents.orchestrator")

AgentName = Literal["hr_agent", "it_agent"]

ROUTING_HISTORY = 10   # messages the router sees
AGENT_HISTORY = 20     # messages the sub-agents see

EMAIL_IN_TEXT = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# Phrase right before an address showing it is the employee's own ("my email is", "I'm", "use", "it's")
SELF_EMAIL_HINT = re.compile(
    r"\b(my (work |correct |actual )?(e-?mail|address|id)( address)?( is)?|i am|i'm|im|this is|use|it's|its|"
    r"reach me at|contact me at)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
IT_KEYWORDS = re.compile(
    r"\b(password|vpn|wi-?fi|laptop|computer|monitor|printer|outlook|mfa|login|log in|software|install|"
    r"licen[cs]e|network|internet|keyboard|mouse|headset|phishing|virus|(?-i:IT))\b",
    re.IGNORECASE,
)
HR_KEYWORDS = re.compile(
    r"\b(leave|vacation|holiday|sick|salary|pay|payslip|payroll|bonus|benefits?|insurance|policy|dress|"
    r"maternity|paternity|onboarding|resign\w*|grievance|expense|reimburse\w*|appraisal|(?-i:HR))\b",
    re.IGNORECASE,
)


class RoutingDecision(BaseModel):
    """Which specialist agents should handle the employee's latest message."""
    agents: list[AgentName] = Field(description="One or both of: hr_agent, it_agent")
    hr_question: str = Field(default="", description=(
        "Only when BOTH agents are selected: the HR part of the latest message, as a self-contained question. "
        "Empty otherwise."))
    it_question: str = Field(default="", description=(
        "Only when BOTH agents are selected: the IT part of the latest message, as a self-contained question. "
        "Empty otherwise."))
    reasoning: str = Field(description="One sentence explaining the routing choice")


router = llm.with_structured_output(RoutingDecision)


def routing_input(messages: list[AnyMessage]) -> list[AnyMessage]:
    """What the LLM router sees: its instructions plus the recent conversation."""
    return [SystemMessage(ORCHESTRATOR_PROMPT), *messages[-ROUTING_HISTORY:]]


def extract_email(text: str) -> str | None:
    match = EMAIL_IN_TEXT.search(text or "")
    return match.group(0).lower() if match else None


def resolve_employee_email(text: str, known: str | None) -> str | None:
    """Return the session's employee email after this message.

    The first email the employee gives is remembered. Later emails only replace it when they are clearly
    the employee's own ("my email is ...", or a message that is just an address) — so mentioning a
    colleague's address doesn't silently switch whose tickets are created or shown.
    """
    mentioned = extract_email(text)
    if not mentioned or not known or mentioned == known:
        return mentioned or known
    preceding = text[: text.lower().find(mentioned)]
    if text.strip().strip(".").lower() == mentioned or SELF_EMAIL_HINT.search(preceding):
        return mentioned
    return known


def fallback_route(text: str) -> RoutingDecision:
    """Keyword routing used when the LLM router is unavailable."""
    agents: list[AgentName] = []
    if HR_KEYWORDS.search(text):
        agents.append("hr_agent")
    if IT_KEYWORDS.search(text):
        agents.append("it_agent")
    return RoutingDecision(agents=agents or ["hr_agent"], reasoning="Keyword fallback (LLM router unavailable).")


def orchestrator(state: SmartDeskState) -> Command[AgentName]:
    messages = state["messages"]
    latest = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    latest_text = latest.text if latest else ""

    # Remember the employee's email for the whole session as soon as they give it;
    # a signed-in employee's verified email is never replaced from chat
    if state.get("email_verified"):
        email = state.get("employee_email")
    else:
        email = resolve_employee_email(latest_text, state.get("employee_email"))

    try:
        decision = router.invoke(routing_input(messages))
    except Exception as e:
        logger.exception("Routing LLM call failed, using keyword fallback: %s", e)
        decision = fallback_route(latest_text)

    agents = list(dict.fromkeys(decision.agents)) or ["hr_agent"]
    logger.info("Routing %r -> %s (%s)", latest_text[:120], agents, decision.reasoning)

    # A mixed HR + IT message is split by the router, so each agent gets only its own part to answer
    parts = {"hr_agent": decision.hr_question.strip(), "it_agent": decision.it_question.strip()}
    split = len(agents) > 1 and all(parts[agent] for agent in agents)
    if split:
        logger.info("Split mixed message: %s", {agent: parts[agent] for agent in agents})

    def task(agent: AgentName) -> AgentTask:
        return {"messages": messages[-AGENT_HISTORY:], "employee_email": email, "question": parts[agent] if split else None}

    return Command(
        update={
            "employee_email": email,
            "selected_agents": agents,
            "routing_reasoning": decision.reasoning,
            "agent_responses": None,  # clear the previous turn's answers
        },
        goto=[Send(agent, task(agent)) for agent in agents],
    )
