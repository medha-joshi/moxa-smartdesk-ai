"""
agents/hr_agent.py — HR agent: answers from the HR knowledge base, raises/checks HR tickets,
and handles greetings, small talk and out-of-scope questions.
"""

from smartDeskAi.agents.prompts import HR_AGENT_EXTRA_RULES
from smartDeskAi.agents.specialist import build_specialist_node
from smartDeskAi.tools.hr_tools import search_hr_knowledge_base

hr_agent = build_specialist_node(
    agent_name="hr_agent",
    domain="HR",
    label="HR",
    search_tool=search_hr_knowledge_base,
    extra_rules=HR_AGENT_EXTRA_RULES,
)
