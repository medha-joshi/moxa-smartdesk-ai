"""
agents/specialist.py — Shared builder for the HR and IT agents.

Each specialist is a LangChain tool-calling agent (its own ReAct loop) with a knowledge-base search
tool plus the Jira ticket tools. It runs as a subgraph inside the parent graph's node, so it inherits
the parent's checkpointer and the ticket-confirmation interrupt() can pause and resume it.
"""

from typing import Callable, Literal

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp

from smartDeskAi.agents.prompts import build_specialist_prompt
from smartDeskAi.config import llm
from smartDeskAi.logger import setup_logger
from smartDeskAi.state import AgentTask
from smartDeskAi.tools.rag import ensure_citations
from smartDeskAi.tools.ticket_tools import AgentContext, make_ticket_tools

AGENT_RECURSION_LIMIT = 25

SPLIT_NOTE = """
## This message
The employee's latest message had parts for both HR and IT, and another assistant answers the other part.
Your part is the employee's last message as shown: answer only that, and don't mention the other part."""


def own_part(messages: list[AnyMessage], question: str | None) -> list[AnyMessage]:
    """Replace the employee's latest message with this agent's part of it, when the router split it."""
    if not question:
        return list(messages)
    last = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=None)
    if last is None:
        return list(messages)
    return [*messages[:last], HumanMessage(question), *messages[last + 1:]]


def build_specialist_node(
    agent_name: str,
    domain: str,
    label: Literal["HR", "IT"],
    search_tool: BaseTool,
    extra_rules: str,
) -> Callable[[AgentTask], dict]:
    logger = setup_logger(f"smartDeskAi.agents.{agent_name}")
    agent = create_agent(
        llm,
        tools=[search_tool, *make_ticket_tools(label)],
        context_schema=AgentContext,
        name=agent_name,
    )

    def run_specialist(task: AgentTask) -> dict:
        email = task.get("employee_email")
        system_prompt = build_specialist_prompt(agent_name, domain, search_tool.name, email, extra_rules)
        question = task.get("question")
        if question:
            system_prompt += SPLIT_NOTE
        try:
            inputs = [SystemMessage(system_prompt), *own_part(task["messages"], question)]
            result = agent.invoke(
                {"messages": inputs},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
                context=AgentContext(employee_email=email),
            )
            response = result["messages"][-1].text.strip()
            tool_results = [m for m in result["messages"][len(inputs):] if isinstance(m, ToolMessage)]
            # Ticket replies are never knowledge-base answers, so only back-fill citations for pure searches
            if tool_results and all(m.name == search_tool.name for m in tool_results):
                response = ensure_citations(response, [m.text for m in tool_results])
        except GraphBubbleUp:
            # interrupt() for ticket confirmation — must propagate to the graph
            raise
        except Exception as e:
            logger.exception("%s failed: %s", agent_name, e)
            response = (
                f"I'm sorry, I ran into a technical problem while looking into your {domain} question. "
                "Please try again in a few moments."
            )
        logger.info("%s response: %s", agent_name, response[:300])
        return {"agent_responses": [{"agent": agent_name, "response": response}]}

    run_specialist.__name__ = agent_name
    return run_specialist
