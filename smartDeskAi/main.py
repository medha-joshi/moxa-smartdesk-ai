"""
main.py — CLI entry point for SmartDeskAiAssistant (Moxa).

    uv run python -m smartDeskAi.main [--reindex] [--verbose] [--graph]
"""

import argparse
import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from smartDeskAi.graph import build_graph
from smartDeskAi.logger import set_console_level, setup_logger
from smartDeskAi.tools.rag import KB_FILES, get_knowledge_base, rebuild_knowledge_bases

logger = setup_logger("smartDeskAi.main")

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
NEW_SESSION_COMMAND = "/new"
GREETING = (
    "Hi, I'm Moxa, Synergy Inc's HR & IT assistant. Ask me about HR policies or IT issues, "
    "raise a support ticket, or check the status of one.\n"
    f"(Type '{NEW_SESSION_COMMAND}' to start a new session, 'exit' to quit.)"
)
TECHNICAL_ERROR = "I'm sorry, something went wrong on my side. Please try again in a moment."


def new_session_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def format_ticket_confirmation(ticket: dict) -> str:
    return (
        "Here is the ticket I will create for you:\n"
        f"  Summary     : {ticket['summary']}\n"
        f"  Description : {ticket['description']}\n"
        f"  Priority    : {ticket['priority']}\n"
        f"  Label       : {ticket['label']}\n"
        f"  Your email  : {ticket['email']}\n"
        "Shall I go ahead? (yes / no — or type extra details to add them to the description)"
    )


def ask_human(interrupt_value: dict) -> str:
    """Collect the employee's answer for a graph interrupt (ticket confirmation)."""
    if interrupt_value.get("type") == "ticket_confirmation":
        print(f"\nMoxa: {format_ticket_confirmation(interrupt_value['ticket'])}")
    else:
        print(f"\nMoxa: {interrupt_value}")
    answer = ""
    while not answer:
        answer = input("You: ").strip()
    return answer


def run_turn(graph: CompiledStateGraph, user_text: str, config: dict) -> str:
    """Run one employee message through the graph, handling any human-in-the-loop interrupts."""
    result = graph.invoke({"messages": [HumanMessage(user_text)]}, config)
    while result.get("__interrupt__"):
        resume = {item.id: ask_human(item.value) for item in result["__interrupt__"]}
        result = graph.invoke(Command(resume=resume), config)
    return result.get("final_response") or TECHNICAL_ERROR


def load_knowledge_bases(reindex: bool) -> None:
    print("Loading knowledge bases...")
    try:
        if reindex:
            rebuild_knowledge_bases()
        for kb_name in KB_FILES:
            get_knowledge_base(kb_name)
    except Exception as e:
        # Keep the chat usable; searches will retry and report the problem politely
        logger.exception("Could not load knowledge bases: %s", e)
        print("Warning: the knowledge bases could not be loaded right now; answers may be limited.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moxa — Synergy Inc's HR & IT help-desk assistant")
    parser.add_argument("--reindex", action="store_true", help="rebuild the Qdrant collections from the JSON knowledge bases")
    parser.add_argument("--verbose", action="store_true", help="show routing and retrieval logs on the console")
    parser.add_argument("--graph", action="store_true", help="print the LangGraph architecture as Mermaid and exit")
    return parser.parse_args()


# ── Entrypoint ──────────────────────────────────────────────

def main():
    """Main entrypoint for the SmartDeskAiAssistant CLI."""
    args = parse_args()
    graph = build_graph()
    if args.graph:
        print(graph.get_graph().draw_mermaid())
        return
    if args.verbose:
        set_console_level(logging.INFO)

    print("Welcome to SmartDeskAiAssistant!")
    load_knowledge_bases(args.reindex)
    config = new_session_config()
    print(f"\nMoxa: {GREETING}")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMoxa: Goodbye! Have a great day.")
            break
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            print("Moxa: Goodbye! Have a great day.")
            break
        if user_text.lower() == NEW_SESSION_COMMAND:
            config = new_session_config()
            print("Moxa: Started a new session. How can I help you?")
            continue

        try:
            reply = run_turn(graph, user_text, config)
        except KeyboardInterrupt:
            reply = "Okay, I've stopped that request."
        except Exception as e:
            logger.exception("Turn failed: %s", e)
            reply = TECHNICAL_ERROR
        print(f"\nMoxa: {reply}")


if __name__ == "__main__":
    main()
