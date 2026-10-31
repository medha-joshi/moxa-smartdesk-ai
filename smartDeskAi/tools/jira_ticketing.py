"""
tools/jira_ticketing.py — Create and read Jira tickets for smartDeskAi.
"""

import re
from functools import lru_cache
from typing import Literal

from jira import JIRA, JIRAError
from smartDeskAi.config import JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY, JIRA_TIMEOUT
from smartDeskAi.logger import setup_logger

logger = setup_logger("smartDeskAi.tools.jira_ticketing")

ISSUE_TYPE = "Bug"

Priority = Literal["Critical", "High", "Medium", "Low"]
# Our Jira has no "Critical" priority; its top level is "Highest"
JIRA_PRIORITY = {"Critical": "Highest", "High": "High", "Medium": "Medium", "Low": "Low"}

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CONFIRM_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "go ahead", "please do"}
CANCEL_WORDS = {"no", "n", "nope", "cancel", "stop", "don't", "dont", "do not", "abort", "never mind", "nevermind"}


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email or ""))


def ticket_url(ticket_key: str) -> str:
    """Browser link the employee can use to track a ticket."""
    return f"{JIRA_SERVER}/browse/{ticket_key}"


@lru_cache(maxsize=1)
def _get_jira() -> JIRA:
    """Connect to Jira Cloud once, on first use."""
    if not (JIRA_SERVER and JIRA_EMAIL and JIRA_API_TOKEN):
        raise RuntimeError("Missing JIRA_SERVER, JIRA_EMAIL or JIRA_API_TOKEN in your .env file")
    # Short timeout and no retries so an unreachable Jira fails fast. Keep get_server_info on: the
    # library needs it to detect Jira Cloud and use the current search API (v2 /search returns 410)
    return JIRA(server=JIRA_SERVER, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN), timeout=JIRA_TIMEOUT, max_retries=0)


class JiraUserNotFoundError(ValueError):
    """The email does not belong to any Jira user."""


def _get_account_id(jira: JIRA, email: str) -> str:
    """Look up a Jira user's accountId by exact email match.

    search_users() is a fuzzy prefix search ("medha_j@yahoo.co" also finds "medha_j@yahoo.com"),
    so only an exact, case-insensitive email match is accepted.
    """
    target = email.strip().lower()
    users = jira.search_users(query=email)
    matches = [u for u in users if (getattr(u, "emailAddress", "") or "").strip().lower() == target]
    if not matches:
        raise JiraUserNotFoundError(f"No Jira user found with email: {email}")
    return matches[0].accountId


def employee_has_jira_account(email: str) -> bool:
    """True if the email belongs to a Jira user (tickets are linked to employees as reporter)."""
    try:
        _get_account_id(_get_jira(), email)
        return True
    except JiraUserNotFoundError:
        return False


def create_ticket(ticket: dict) -> str:
    """Create a Jira ticket.

    Args:
        ticket: dict with keys 'email', 'summary', 'description', 'label', 'priority'

    Returns:
        The created ticket number (e.g. 'IKCS-12'), or an error message starting with 'Error:'.
    """
    missing = [k for k in ("email", "summary", "description", "label", "priority") if not ticket.get(k)]
    if missing:
        return f"Error: missing ticket fields: {', '.join(missing)}"
    if not is_valid_email(ticket["email"]):
        return f"Error: invalid email address: {ticket['email']}"

    try:
        jira = _get_jira()
        account_id = _get_account_id(jira, ticket["email"])
        issue_data = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": ticket["summary"],
            "description": f"{ticket['description']}\n\nReported by: {ticket['email']}",
            "issuetype": {"name": ISSUE_TYPE},
            "priority": {"name": JIRA_PRIORITY.get(ticket["priority"], ticket["priority"])},
            "labels": [ticket["label"]],
            "reporter": {"id": account_id},
        }
        new_issue = jira.create_issue(fields=issue_data)
        logger.info("Created Jira ticket %s for %s", new_issue.key, ticket["email"])
        return new_issue.key
    except JIRAError as e:
        logger.error("Jira error creating ticket: %s", e.text)
        return f"Error: {e.text}"
    except Exception as e:
        logger.error("Error creating ticket: %s", e)
        return f"Error: {e}"


def read_tickets(email: str) -> list[str]:
    """Return the ticket numbers reported by the given email, newest first."""
    jira = _get_jira()
    account_id = _get_account_id(jira, email)
    jql_query = f'project = {JIRA_PROJECT_KEY} AND reporter = "{account_id}" ORDER BY created DESC'
    issues = jira.search_issues(jql_query, maxResults=False, fields="key")
    return [issue.key for issue in issues]


def get_tickets(email: str, label: str | None = None, max_results: int = 20) -> list[dict]:
    """Return a short status summary of the tickets reported by the given email, newest first.

    If label is given, only tickets with that label are returned.
    """
    jira = _get_jira()
    account_id = _get_account_id(jira, email)
    label_filter = f' AND labels = "{label}"' if label else ""
    jql_query = f'project = {JIRA_PROJECT_KEY} AND reporter = "{account_id}"{label_filter} ORDER BY created DESC'
    issues = jira.search_issues(jql_query, maxResults=max_results, fields="summary,status,priority,labels,updated")
    return [
        {
            "key": issue.key,
            "summary": issue.fields.summary,
            "status": issue.fields.status.name,
            "is_open": issue.fields.status.statusCategory.key != "done",
            "priority": issue.fields.priority.name if issue.fields.priority else "",
            "labels": issue.fields.labels,
            "updated": issue.fields.updated[:10],
        }
        for issue in issues
    ]


def get_ticket_details(ticket_key: str, email: str, max_comments: int = 5) -> dict:
    """Return status, description and latest comments for one ticket.

    Raises ValueError if the ticket was not reported by the given email.
    """
    jira = _get_jira()
    account_id = _get_account_id(jira, email)
    issue = jira.issue(ticket_key, fields="summary,description,status,priority,labels,created,updated,reporter,comment")
    fields = issue.fields
    # Only show employees their own tickets
    if not fields.reporter or fields.reporter.accountId != account_id:
        raise ValueError(f"Ticket {ticket_key} was not raised by {email}")

    comments = fields.comment.comments[-max_comments:]
    return {
        "key": issue.key,
        "summary": fields.summary,
        "description": fields.description or "",
        "status": fields.status.name,
        "priority": fields.priority.name if fields.priority else "",
        "labels": fields.labels,
        "created": fields.created[:10],
        "updated": fields.updated[:10],
        "comments": [
            {"author": c.author.displayName, "created": c.created[:10], "body": c.body}
            for c in comments
        ],
    }


