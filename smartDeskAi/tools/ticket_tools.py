"""
tools/ticket_tools.py — LangChain tools for Jira ticket creation (with human-in-the-loop) and status checks.
"""

import os
import re
from dataclasses import dataclass
from typing import Literal

from jira import JIRAError
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout

from smartDeskAi.logger import setup_logger
from smartDeskAi.tools.jira_ticketing import (
    CANCEL_WORDS, CONFIRM_WORDS, JiraUserNotFoundError, Priority, create_ticket, employee_has_jira_account,
    get_ticket_details, get_tickets, ticket_url,
)

logger = setup_logger("smartDeskAi.tools.ticket_tools")

EMAIL_REQUIRED = (
    "Error: EMAIL_REQUIRED — the employee's email address is not known yet. "
    "Ask the employee for their work email address before continuing."
)
JIRA_UNAVAILABLE = (
    "Error: JIRA_UNAVAILABLE — the ticketing system could not be reached. "
    "Apologise politely and suggest the employee try again later."
)

TICKET_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]*-\d+")

# Words that make a reply a yes / no, and words that may surround them without adding information
_AFFIRM = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "proceed",
           "absolutely", "definitely", "correct", "perfect", "great", "fine", "submit"}
_NEGATE = {"no", "n", "nope", "nah", "cancel", "abort", "stop", "don't", "dont", "not", "never"}
_FILLER = {"please", "go", "ahead", "do", "it", "create", "raise", "file", "submit", "thanks", "thank", "you",
           "i", "that's", "thats", "that", "sounds", "good", "looks", "is", "all", "right", "alright", "now",
           "the", "ticket", "this", "one", "thing", "for", "it's", "its", "need", "want", "anymore", "mind",
           "never", "a", "let's", "lets", "just"}
_APPROVAL_FILLER = {"go", "do", "create", "submit", "raise", "file", "good", "alright", "right"}
_PRIORITY_VALUE = r"(?P<{name}>low|medium|high|critical)"
# "priority high", "priority: critical", "change the priority to low" / "make it high priority", "low prio"
_PRIORITY_CHANGE = re.compile(
    r"(?:\b(?:set|change|make|update|raise|lower)?\s*(?:the\s+|it\s+)?(?:to\s+)?)?"
    r"(?:\bpriority\b\s*(?:to|as|=|:|-|is|should\s+be)?\s*" + _PRIORITY_VALUE.format(name="after") + r"\b"
    r"|\b" + _PRIORITY_VALUE.format(name="before") + r"\s+prio(?:rity)?\b)",
    re.IGNORECASE,
)
# "details: <text>" / "details <text>" / "add details: <text>"
_DETAILS_KEYWORD = re.compile(r"\b(?:add(?:itional)?\s+)?details?\b\s*[:\-]?\s*", re.IGNORECASE)
_LEADING_CONNECTORS = re.compile(
    r"^(?:[\s,.;:\-]*\b(?:yes|yeah|yep|ok|okay|sure|and|also|but|so|please|actually|wait|plus)\b)*", re.IGNORECASE
)
# Words left over after removing a priority change that carry no detail of their own
_EDIT_FILLER = _FILLER | _AFFIRM | {"and", "also", "but", "please", "make", "set", "change", "update", "to",
                                    "as", "priority", "prio", "instead", "actually", "can", "could", "you", "wait"}


@dataclass
class AgentContext:
    """Per-run context handed to the HR/IT agents (read by tools through ToolRuntime)."""
    employee_email: str | None = None


def _normalize(answer: str) -> str:
    return re.sub(r"[^\w\s']", " ", (answer or "").lower()).strip()


def classify_confirmation(answer: str) -> Literal["confirm", "cancel", "details"]:
    """Interpret the employee's reply to 'Shall I go ahead?'.

    A short reply made only of yes-words (or no-words) plus filler is a decision; anything carrying
    other content (e.g. "no wait, add that my laptop is new", "priority high") is treated as edits to
    the ticket — see parse_ticket_edits().
    """
    words = _normalize(answer).split()
    text = " ".join(words)
    if text in CONFIRM_WORDS:
        return "confirm"
    if text in CANCEL_WORDS:
        return "cancel"
    meaningful = [w for w in words if w not in _FILLER]
    if meaningful and all(w in _NEGATE for w in meaningful):
        return "cancel"
    if meaningful and all(w in _AFFIRM for w in meaningful):
        return "confirm"
    if not meaningful and _APPROVAL_FILLER.intersection(words):
        return "confirm"  # "go ahead", "do it", "looks good, submit it"
    return "details"


def parse_ticket_edits(answer: str) -> tuple[Priority | None, str | None]:
    """Split a reply to the ticket confirmation into a priority override and extra description details.

    Accepts the structured form ("priority high details it fails every morning") as well as free text
    ("make it critical priority, the whole team is blocked"). Details keep the employee's original wording.
    """
    text = (answer or "").strip()
    priority = None
    match = _PRIORITY_CHANGE.search(text)
    if match:
        priority = (match.group("after") or match.group("before")).capitalize()
        text = (text[: match.start()] + " " + text[match.end():]).strip()

    keyword = _DETAILS_KEYWORD.search(text)
    if keyword:
        details = text[keyword.end():]
    else:
        details = text
        if priority:
            # Trim framing words left in front of the removed priority phrase ("yes but , the team is blocked")
            details = _LEADING_CONNECTORS.sub("", details)
    details = details.strip(" ,.;:-")
    # Drop leftovers such as "yes, and also" that only framed the priority change
    if not [w for w in _normalize(details).split() if w not in _EDIT_FILLER]:
        details = ""
    return priority, details or None


# Words too common to say whether a ticket is about a topic
_TOPIC_STOPWORDS = {"the", "and", "for", "with", "not", "my", "any", "ticket", "tickets", "issue", "issues",
                    "problem", "about", "from", "can", "cannot", "can't", "unable", "working", "work", "have",
                    "has", "had", "get", "got", "this", "that", "was", "were", "is", "are", "of", "to", "on", "in"}


def _topic_words(text: str) -> set[str]:
    return {w for w in _normalize(text).split() if len(w) >= 3 and w not in _TOPIC_STOPWORDS}


def _same_stem(a: str, b: str) -> bool:
    """Crude stemming: "vpn"/"vpns", "printing"/"printer" share a stem, "mouse"/"mouth" don't."""
    common = len(os.path.commonprefix([a, b]))
    return common == min(len(a), len(b)) or common >= max(4, min(len(a), len(b)) - 3)


def _topic_score(query_words: set[str], summary: str) -> int:
    """Number of query words whose stem appears in the summary."""
    summary_words = _topic_words(summary)
    return sum(any(_same_stem(q, s) for s in summary_words) for q in query_words)


def _format_ticket_line(t: dict) -> str:
    return (f"- {t['key']} | {t['summary']} | status: {t['status']} | {'open' if t['is_open'] else 'closed'} | "
            f"priority: {t['priority']} | labels: {', '.join(t['labels']) or 'none'} | updated {t['updated']}")


def _jira_error_message(error: Exception) -> str:
    if isinstance(error, (RequestsConnectionError, Timeout)):
        return JIRA_UNAVAILABLE
    if isinstance(error, ValueError):  # unknown Jira user / ticket owned by someone else
        return f"Error: {error}. Ask the employee to double-check the details."
    if isinstance(error, JIRAError) and error.status_code in (400, 404, 405):
        return "Error: that ticket does not exist. Ask the employee to double-check the ticket ID."
    if isinstance(error, RuntimeError) or (isinstance(error, JIRAError) and error.status_code in (401, 403)):
        return "Error: the ticketing system is not configured. Apologise and suggest contacting the service desk directly."
    return JIRA_UNAVAILABLE


def make_ticket_tools(label: Literal["HR", "IT"]) -> list[BaseTool]:
    """Build ticket tools for one agent; tickets it creates are auto-labelled with `label`."""

    @tool
    def create_support_ticket(
        summary: str,
        description: str,
        runtime: ToolRuntime[AgentContext],
        priority: Priority = "Medium",
    ) -> str:
        """Create a Jira support ticket after the employee has agreed to raise one.

        The employee is shown the full ticket and must explicitly confirm before it is created;
        they may also add more details at that point.

        Args:
            summary: Short one-line title, e.g. "Unable to connect to VPN from home".
            description: Detailed description summarising the conversation so far: the problem,
                what the employee already tried, and any relevant details they shared. Refer to
                them as "the employee" — never infer pronouns from a name.
            priority: Low, Medium, High or Critical, inferred from urgency and business impact. But also allow the employee to override it if they explicitly say "make it high priority" or similar.
        """
        email = runtime.context.employee_email if runtime.context else None
        if not email:
            return EMAIL_REQUIRED

        # Check the employee is known to Jira before asking them to confirm anything
        try:
            if not employee_has_jira_account(email):
                return (f"Error: UNKNOWN_EMAIL — {email} is not registered in the ticketing system. "
                        "Ask the employee to double-check their work email address and provide the correct one.")
        except Exception as e:
            logger.exception("Could not verify Jira user %s", email)
            return _jira_error_message(e)

        ticket = {"email": email, "summary": summary.strip(), "description": description.strip(),
                  "label": label, "priority": priority}

        # Human-in-the-loop: nothing is created until the employee explicitly says yes
        while True:
            answer = interrupt({"type": "ticket_confirmation", "ticket": dict(ticket)})
            if isinstance(answer, dict):
                # Structured edits from the web UI form: {"priority": ..., "details": ...}
                new_priority, details = answer.get("priority"), answer.get("details")
            else:
                decision = classify_confirmation(str(answer))
                if decision == "confirm":
                    break
                if decision == "cancel":
                    logger.info("Employee %s cancelled ticket creation", email)
                    return "The employee declined. The ticket was NOT created. Acknowledge this politely."
                # Anything else edits the ticket, which is then shown again for confirmation
                new_priority, details = parse_ticket_edits(str(answer))
            if new_priority:
                ticket["priority"] = new_priority
                logger.info("Employee %s changed ticket priority to %s", email, new_priority)
            if details:
                ticket["description"] += f"\n\nAdditional details from the employee: {details}"
                logger.info("Employee %s added ticket details: %s", email, details)

        result = create_ticket(ticket)
        if result.startswith("Error:"):
            logger.error("Ticket creation failed for %s: %s", email, result)
            return (f"Error: the ticket could not be created ({result.removeprefix('Error:').strip()}). "
                    "Apologise politely and suggest the employee try again later.")
        # The employee may have changed the priority/details while confirming, so these are the final values
        return (f"Ticket created successfully. Ticket ID: {result}. Link: {ticket_url(result)}. "
                f"Summary: {ticket['summary']}. Priority: {ticket['priority']}. Label: {label}. "
                "These are the ticket's final values as confirmed by the employee; report them exactly, "
                "even if they differ from the arguments you passed.")

    @tool
    def list_my_tickets(runtime: ToolRuntime[AgentContext]) -> str:
        """List the support tickets the current employee has raised (newest first) with their status."""
        email = runtime.context.employee_email if runtime.context else None
        if not email:
            return EMAIL_REQUIRED
        try:
            tickets = get_tickets(email)
        except JiraUserNotFoundError:
            return (f"NO_TICKETS: no tickets were found for {email}. Tell the employee clearly, and suggest "
                    "they double-check the email address in case of a typo.")
        except Exception as e:
            logger.exception("Could not list tickets for %s", email)
            return _jira_error_message(e)
        if not tickets:
            return f"NO_TICKETS: no tickets were found for {email}. Tell the employee clearly."
        lines = [_format_ticket_line(t) for t in tickets]
        open_count = sum(t["is_open"] for t in tickets)
        return f"{len(tickets)} ticket(s) found for {email} ({open_count} open):\n" + "\n".join(lines)

    @tool
    def get_my_tickets_by_topic(summary: str, runtime: ToolRuntime[AgentContext]) -> str:
        """Find the current employee's tickets about a topic (best matches first) with their status.

        Use when the employee asks about a ticket by what it is about rather than by ticket ID,
        e.g. "what happened to my VPN ticket?".

        Args:
            summary: A few words describing the topic, e.g. "VPN connection" or "laptop battery".
        """
        email = runtime.context.employee_email if runtime.context else None
        if not email:
            return EMAIL_REQUIRED
        query_words = _topic_words(summary)
        if not query_words:
            return "Error: that topic is too vague. Ask the employee what the ticket was about."
        try:
            tickets = get_tickets(email, max_results=100)
        except JiraUserNotFoundError:
            return (f"NO_TICKETS: no tickets were found for {email}. Tell the employee clearly, and suggest "
                    "they double-check the email address in case of a typo.")
        except Exception as e:
            logger.exception("Could not search tickets for %s", email)
            return _jira_error_message(e)
        scored = [(_topic_score(query_words, t["summary"]), t) for t in tickets]
        # sorted() is stable, so equally good matches stay newest first
        matches = [t for score, t in sorted(scored, key=lambda st: -st[0]) if score]
        if not matches:
            return (f"NO_MATCHING_TICKETS: none of the {len(tickets)} ticket(s) for {email} match "
                    f"\"{summary}\". Tell the employee clearly; offer to list all their tickets.")
        return (f"{len(matches)} ticket(s) for {email} match \"{summary}\":\n"
                + "\n".join(_format_ticket_line(t) for t in matches))

    @tool
    def get_ticket_status(ticket_key: str, runtime: ToolRuntime[AgentContext]) -> str:
        """Get the full status, description and latest support-team comments for one ticket.

        Args:
            ticket_key: The Jira ticket ID, e.g. "IKCS-12".
        """
        email = runtime.context.employee_email if runtime.context else None
        if not email:
            return EMAIL_REQUIRED
        match = TICKET_KEY.search(ticket_key or "")
        if not match:
            return "Error: that ticket ID looks invalid. Ask the employee to double-check it (e.g. IKCS-12)."
        try:
            details = get_ticket_details(match.group(0).upper(), email)
        except Exception as e:
            logger.exception("Could not fetch ticket %s for %s", ticket_key, email)
            return _jira_error_message(e)
        comments = "\n".join(
            f"  - {c['author']} ({c['created']}): {c['body']}" for c in details["comments"]
        ) or "  (no comments from the support team yet)"
        return (
            f"{details['key']}: {details['summary']}\n"
            f"Status: {details['status']} | Priority: {details['priority']} | Labels: {', '.join(details['labels']) or 'none'}\n"
            f"Created: {details['created']} | Last updated: {details['updated']}\n"
            f"Link: {ticket_url(details['key'])}\n"
            f"Description: {details['description']}\n"
            f"Support-team comments (ALWAYS include these in your reply, or say there are no updates yet):\n{comments}"
        )

    return [create_support_ticket, list_my_tickets, get_my_tickets_by_topic, get_ticket_status]
