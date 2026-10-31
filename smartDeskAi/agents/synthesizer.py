"""
agents/synthesizer.py — Merges the parallel HR/IT answers into one reply for the employee.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smartDeskAi.agents.prompts import SYNTHESIZER_PROMPT
from smartDeskAi.config import llm
from smartDeskAi.logger import setup_logger
from smartDeskAi.state import SmartDeskState

logger = setup_logger("smartDeskAi.agents.synthesizer")

AGENT_ORDER = {"hr_agent": 0, "it_agent": 1}
AGENT_HEADINGS = {"hr_agent": "HR", "it_agent": "IT"}
NO_RESPONSE = "I'm sorry, I wasn't able to put together an answer just now. Could you please try again?"


def synthesizer(state: SmartDeskState) -> dict:
    responses = sorted(state.get("agent_responses") or [], key=lambda r: AGENT_ORDER.get(r["agent"], 99))

    if not responses:
        final = NO_RESPONSE
    elif len(responses) == 1:
        # Nothing to merge — pass the single answer through untouched
        final = responses[0]["response"]
    else:
        latest = next((m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
        replies = "\n\n".join(f"--- {AGENT_HEADINGS[r['agent']]} reply ---\n{r['response']}" for r in responses)
        try:
            final = llm.invoke([
                SystemMessage(SYNTHESIZER_PROMPT),
                HumanMessage(f"Employee's message:\n{latest}\n\nSpecialist replies:\n{replies}"),
            ]).text.strip()
        except Exception as e:
            logger.exception("Synthesizer LLM call failed, concatenating replies: %s", e)
            final = "\n\n".join(f"{AGENT_HEADINGS[r['agent']]}:\n{r['response']}" for r in responses)

    return {"messages": [AIMessage(final, name="moxa")], "final_response": final}
