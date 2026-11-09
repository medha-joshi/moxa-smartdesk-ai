from types import SimpleNamespace

import pytest
from jira import JIRAError
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout

from smartDeskAi.tools import ticket_tools
from smartDeskAi.tools.jira_ticketing import JiraUserNotFoundError
from smartDeskAi.tools.ticket_tools import (
    EMAIL_REQUIRED, JIRA_UNAVAILABLE, AgentContext, _jira_error_message, _same_stem, _topic_score, _topic_words,
    classify_confirmation, make_ticket_tools, parse_ticket_edits,
)

EMAIL = "asha@synergy.com"


# ---------- confirmation replies ----------

@pytest.mark.parametrize("answer", [
    "yes", "Y", "ok", "Yes please, go ahead!", "go ahead", "do it", "looks good, submit it", "sure thing",
    "Yep, that's perfect", "confirmed",
])
def test_classify_confirm(answer):
    assert classify_confirmation(answer) == "confirm"


@pytest.mark.parametrize("answer", [
    "no", "No thanks", "cancel", "never mind", "don't create it", "nope, not anymore", "stop",
])
def test_classify_cancel(answer):
    assert classify_confirmation(answer) == "cancel"


@pytest.mark.parametrize("answer", [
    "priority high", "no wait, add that my laptop is brand new", "details: it fails every morning",
    "make it critical", "",
])
def test_classify_details(answer):
    assert classify_confirmation(answer) == "details"


@pytest.mark.parametrize("answer, expected", [
    ("priority high", ("High", None)),
    ("Priority: CRITICAL", ("Critical", None)),
    ("change the priority to low", ("Low", None)),
    ("low prio", ("Low", None)),
    ("priority medium details it fails every morning", ("Medium", "it fails every morning")),
    ("make it critical priority, the whole team is blocked", ("Critical", "the whole team is blocked")),
    ("yes, and also high priority", ("High", None)),
    ("add details: only happens on office Wi-Fi", (None, "only happens on office Wi-Fi")),
    ("the screen also flickers", (None, "the screen also flickers")),
    ("", (None, None)),
])
def test_parse_ticket_edits(answer, expected):
    assert parse_ticket_edits(answer) == expected


# ---------- topic matching ----------

def test_topic_words_drop_stopwords_and_short_words():
    assert _topic_words("My VPN ticket is not working!") == {"vpn"}


@pytest.mark.parametrize("a, b, same", [
    ("vpn", "vpns", True), ("printing", "printer", True), ("laptop", "laptop", True),
    ("mouse", "mouth", False), ("battery", "badge", False),
])
def test_same_stem(a, b, same):
    assert _same_stem(a, b) is same


def test_topic_score_counts_matching_query_words():
    assert _topic_score({"vpn", "connection"}, "VPN connections drop at home") == 2
    assert _topic_score({"printer"}, "Laptop battery drains") == 0


# ---------- Jira error messages ----------

@pytest.mark.parametrize("error, expected", [
    (RequestsConnectionError(), JIRA_UNAVAILABLE),
    (Timeout(), JIRA_UNAVAILABLE),
    (JIRAError(status_code=500), JIRA_UNAVAILABLE),
    (OSError("boom"), JIRA_UNAVAILABLE),
])
def test_jira_error_unavailable(error, expected):
    assert _jira_error_message(error) == expected


def test_jira_error_specific_messages():
    assert "double-check the details" in _jira_error_message(ValueError("not yours"))
    assert "does not exist" in _jira_error_message(JIRAError(status_code=404))
    assert "not configured" in _jira_error_message(JIRAError(status_code=401))
    assert "not configured" in _jira_error_message(RuntimeError("Missing JIRA_SERVER"))


# ---------- tools ----------

def runtime(email: str | None = EMAIL):
    return SimpleNamespace(context=AgentContext(employee_email=email))


@pytest.fixture
def tools():
    create, list_tickets, by_topic, status = make_ticket_tools("IT")
    return SimpleNamespace(create=create.func, list=list_tickets.func, by_topic=by_topic.func, status=status.func)


def ticket(key, summary, is_open=True):
    return {"key": key, "summary": summary, "status": "Open" if is_open else "Done", "is_open": is_open,
            "priority": "Medium", "labels": ["IT"], "updated": "2026-11-01"}


def test_tool_names_and_label():
    assert [t.name for t in make_ticket_tools("HR")] == [
        "create_support_ticket", "list_my_tickets", "get_my_tickets_by_topic", "get_ticket_status",
    ]


def test_every_tool_requires_email(tools):
    no_email = runtime(None)
    assert tools.create(summary="s", description="d", runtime=no_email) == EMAIL_REQUIRED
    assert tools.list(runtime=no_email) == EMAIL_REQUIRED
    assert tools.by_topic(summary="vpn", runtime=no_email) == EMAIL_REQUIRED
    assert tools.status(ticket_key="IKCS-1", runtime=no_email) == EMAIL_REQUIRED


def test_create_rejects_email_unknown_to_jira(tools, monkeypatch):
    monkeypatch.setattr(ticket_tools, "employee_has_jira_account", lambda email: False)
    result = tools.create(summary="s", description="d", runtime=runtime())
    assert result.startswith("Error: UNKNOWN_EMAIL") and EMAIL in result


def test_create_reports_jira_outage_before_asking_to_confirm(tools, monkeypatch):
    def down(email):
        raise RequestsConnectionError()
    monkeypatch.setattr(ticket_tools, "employee_has_jira_account", down)
    assert tools.create(summary="s", description="d", runtime=runtime()) == JIRA_UNAVAILABLE


def test_list_my_tickets(tools, monkeypatch):
    monkeypatch.setattr(ticket_tools, "get_tickets", lambda email: [
        ticket("IKCS-2", "VPN drops", is_open=True), ticket("IKCS-1", "Printer jam", is_open=False),
    ])
    result = tools.list(runtime=runtime())
    assert result.startswith(f"2 ticket(s) found for {EMAIL} (1 open)")
    assert "- IKCS-2 | VPN drops | status: Open | open" in result
    assert "- IKCS-1 | Printer jam | status: Done | closed" in result


@pytest.mark.parametrize("outcome", [[], JiraUserNotFoundError("nobody")])
def test_list_my_tickets_none_found(tools, monkeypatch, outcome):
    def get_tickets(email):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(ticket_tools, "get_tickets", get_tickets)
    assert tools.list(runtime=runtime()).startswith("NO_TICKETS")


def test_tickets_by_topic_ranks_best_match_first(tools, monkeypatch):
    monkeypatch.setattr(ticket_tools, "get_tickets", lambda email, max_results: [
        ticket("IKCS-3", "Laptop battery drains fast"),
        ticket("IKCS-2", "VPN connection drops"),
        ticket("IKCS-1", "VPN slow at home"),
    ])
    result = tools.by_topic(summary="VPN connection", runtime=runtime())
    lines = result.splitlines()
    assert lines[0] == f'2 ticket(s) for {EMAIL} match "VPN connection":'
    assert lines[1].startswith("- IKCS-2") and lines[2].startswith("- IKCS-1")


def test_tickets_by_topic_no_match_and_vague(tools, monkeypatch):
    monkeypatch.setattr(ticket_tools, "get_tickets", lambda email, max_results: [ticket("IKCS-1", "Printer jam")])
    assert tools.by_topic(summary="payroll", runtime=runtime()).startswith("NO_MATCHING_TICKETS")
    assert "too vague" in tools.by_topic(summary="my ticket", runtime=runtime())


def test_ticket_status_formats_details_and_normalizes_key(tools, monkeypatch):
    seen = {}

    def details(key, email):
        seen["key"] = key
        return {"key": key, "summary": "VPN drops", "description": "Drops hourly", "status": "In Progress",
                "priority": "High", "labels": ["IT"], "created": "2026-10-30", "updated": "2026-11-02",
                "comments": [{"author": "Service Desk", "created": "2026-11-02", "body": "Looking into it"}]}
    monkeypatch.setattr(ticket_tools, "get_ticket_details", details)

    result = tools.status(ticket_key="what about ikcs-12?", runtime=runtime())

    assert seen["key"] == "IKCS-12"
    assert "IKCS-12: VPN drops" in result
    assert "Link: https://jira.example.test/browse/IKCS-12" in result
    assert "- Service Desk (2026-11-02): Looking into it" in result


def test_ticket_status_invalid_key_and_errors(tools, monkeypatch):
    assert "looks invalid" in tools.status(ticket_key="twelve", runtime=runtime())

    def not_found(key, email):
        raise JIRAError(status_code=404)
    monkeypatch.setattr(ticket_tools, "get_ticket_details", not_found)
    assert "does not exist" in tools.status(ticket_key="IKCS-404", runtime=runtime())
