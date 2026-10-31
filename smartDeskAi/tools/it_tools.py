"""
tools/it_tools.py — Search the IT knowledge base.
"""

from langchain_core.tools import tool

from smartDeskAi.logger import setup_logger
from smartDeskAi.tools.rag import format_search_results, get_knowledge_base

logger = setup_logger("smartDeskAi.tools.it_tools")


@tool
def search_it_knowledge_base(query: str) -> str:
    """Search Synergy Inc's IT knowledge base.

    Covers access & accounts (passwords, MFA, permissions), hardware (laptops, monitors, peripherals),
    network & connectivity (VPN, Wi-Fi), security & policy, and software & applications.

    Args:
        query: The employee's question as a complete, self-contained sentence (e.g. "Why is my VPN not connecting from home?").
    """
    try:
        chunks = get_knowledge_base("it").search(query)
    except Exception as e:
        logger.exception("IT knowledge base search failed: %s", e)
        return "Error: the IT knowledge base is temporarily unavailable. Apologise and suggest trying again later."
    return format_search_results("IT", query, chunks)
