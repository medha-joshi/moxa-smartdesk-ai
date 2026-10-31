"""
agents/specialist.py — Shared builder for the HR and IT agents.

Each specialist is a LangChain tool-calling agent (its own ReAct loop) with a knowledge-base search
tool plus the Jira ticket tools. It runs as a subgraph inside the parent graph's node, so it inherits
the parent's checkpointer and the ticket-confirmation interrupt() can pause and resume it.
"""

from typing import Callable, Literal

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp

from smartDeskAi.agents.prompts import build_specialist_prompt
from smartDeskAi.config import llm
from smartDeskAi.logger import setup_logger
from smartDeskAi.state import AgentTask
from smartDeskAi.tools.ticket_tools import AgentContext, make_ticket_tools

AGENT_RECURSION_LIMIT = 25


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
        try:
            result = agent.invoke(
                {"messages": [SystemMessage(system_prompt), *task["messages"]]},
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
                context=AgentContext(employee_email=email),
            )
            response = result["messages"][-1].text.strip()
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
