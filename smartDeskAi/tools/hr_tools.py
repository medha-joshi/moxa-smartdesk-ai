"""
tools/hr_tools.py — Search the HR knowledge base.
"""

from langchain_core.tools import tool

from smartDeskAi.logger import setup_logger
from smartDeskAi.tools.rag import format_search_results, get_knowledge_base

logger = setup_logger("smartDeskAi.tools.hr_tools")


@tool
def search_hr_knowledge_base(query: str) -> str:
    """Search Synergy Inc's HR knowledge base.

    Covers compensation & benefits, leave & attendance, workplace policies & conduct, expenses &
    reimbursement, performance management, onboarding, recruitment & referrals, internships,
    equal opportunity & diversity, and separation & exit.

    Args:
        query: The employee's question as a complete, self-contained sentence (e.g. "How many annual leave days do I get?").
    """
    try:
        chunks = get_knowledge_base("hr").search(query)
    except Exception as e:
        logger.exception("HR knowledge base search failed: %s", e)
        return "Error: the HR knowledge base is temporarily unavailable. Apologise and suggest trying again later."
    return format_search_results("HR", query, chunks)
