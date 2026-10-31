"""
agents/it_agent.py — IT agent: answers from the IT knowledge base and raises/checks IT tickets.
"""

from smartDeskAi.agents.prompts import IT_AGENT_EXTRA_RULES
from smartDeskAi.agents.specialist import build_specialist_node
from smartDeskAi.tools.it_tools import search_it_knowledge_base

it_agent = build_specialist_node(
    agent_name="it_agent",
    domain="IT",
    label="IT",
    search_tool=search_it_knowledge_base,
    extra_rules=IT_AGENT_EXTRA_RULES,
)
