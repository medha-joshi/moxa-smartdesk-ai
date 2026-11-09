from types import SimpleNamespace as NS

import pytest
from jira import JIRAError

from smartDeskAi.tools import jira_ticketing as jt

EMAIL = "asha@synergy.com"
VALID_TICKET = {"email": EMAIL, "summary": "VPN down", "description": "Can't connect", "label": "IT",
                "priority": "Critical"}


class FakeJira:
    """Just enough of jira.JIRA for jira_ticketing."""

    def __init__(self, users=None, issues=None, issue=None, create_error=None):
        self.users = users if users is not None else [NS(emailAddress=EMAIL, accountId="acc-asha")]
        self.issues = issues or []
        self._issue = issue
        self.create_error = create_error
        self.created = []
        self.searches = []

    def search_users(self, query):
        return self.users

    def create_issue(self, fields):
        if self.create_error:
            raise self.create_error
        self.created.append(fields)
        return NS(key="IKCS-42")

    def search_issues(self, jql, maxResults, fields):
        self.searches.append(jql)
        return self.issues

    def issue(self, key, fields):
        return self._issue


@pytest.fixture
def use_jira(monkeypatch):
    def install(fake: FakeJira) -> FakeJira:
        monkeypatch.setattr(jt, "_get_jira", lambda: fake)
        return fake
    return install


@pytest.mark.parametrize("email, valid", [
    ("a@b.co", True), ("first.last@synergy.com", True), ("no-at-sign.com", False), ("a@b", False),
    ("a b@c.com", False), ("", False), (None, False),
])
def test_is_valid_email(email, valid):
    assert jt.is_valid_email(email) is valid


def test_ticket_url():
    assert jt.ticket_url("IKCS-1") == "https://jira.example.test/browse/IKCS-1"


def test_account_lookup_requires_exact_email_match():
    fuzzy = FakeJira(users=[NS(emailAddress="asha@synergy.com.au", accountId="other"),
                            NS(emailAddress="ASHA@Synergy.com", accountId="acc-asha")])
    assert jt._get_account_id(fuzzy, EMAIL) == "acc-asha"
    with pytest.raises(jt.JiraUserNotFoundError):
        jt._get_account_id(FakeJira(users=[NS(emailAddress="asha@synergy.co", accountId="x")]), EMAIL)


def test_employee_has_jira_account(use_jira):
    use_jira(FakeJira())
    assert jt.employee_has_jira_account(EMAIL)
    use_jira(FakeJira(users=[]))
    assert not jt.employee_has_jira_account(EMAIL)


def test_create_ticket_validates_before_calling_jira(use_jira):
    fake = use_jira(FakeJira())
    assert jt.create_ticket({**VALID_TICKET, "summary": ""}) == "Error: missing ticket fields: summary"
    assert jt.create_ticket({**VALID_TICKET, "email": "bad"}) == "Error: invalid email address: bad"
    assert fake.created == []


def test_create_ticket_builds_issue(use_jira):
    fake = use_jira(FakeJira())
    assert jt.create_ticket(VALID_TICKET) == "IKCS-42"
    [fields] = fake.created
    assert fields["project"] == {"key": "IKCS"}
    assert fields["priority"] == {"name": "Highest"}  # Critical is mapped to Jira's top priority
    assert fields["labels"] == ["IT"]
    assert fields["reporter"] == {"id": "acc-asha"}
    assert fields["description"].endswith(f"Reported by: {EMAIL}")


def test_create_ticket_returns_errors_instead_of_raising(use_jira):
    use_jira(FakeJira(create_error=JIRAError(status_code=400, text="Field 'priority' is invalid")))
    assert jt.create_ticket(VALID_TICKET) == "Error: Field 'priority' is invalid"
    use_jira(FakeJira(users=[]))
    assert jt.create_ticket(VALID_TICKET).startswith("Error: No Jira user found")


def jira_issue(key, done=False, priority="High"):
    return NS(key=key, fields=NS(
        summary=f"summary {key}", labels=["IT"], updated="2026-11-02T10:00:00.000+0000",
        status=NS(name="Done" if done else "Open", statusCategory=NS(key="done" if done else "new")),
        priority=NS(name=priority) if priority else None,
    ))


def test_get_tickets_maps_fields_and_filters_by_label(use_jira):
    fake = use_jira(FakeJira(issues=[jira_issue("IKCS-2"), jira_issue("IKCS-1", done=True, priority=None)]))
    tickets = jt.get_tickets(EMAIL, label="IT")
    assert tickets[0] == {"key": "IKCS-2", "summary": "summary IKCS-2", "status": "Open", "is_open": True,
                          "priority": "High", "labels": ["IT"], "updated": "2026-11-02"}
    assert tickets[1]["is_open"] is False and tickets[1]["priority"] == ""
    assert 'reporter = "acc-asha" AND labels = "IT"' in fake.searches[0]


def test_read_tickets(use_jira):
    use_jira(FakeJira(issues=[NS(key="IKCS-2"), NS(key="IKCS-1")]))
    assert jt.read_tickets(EMAIL) == ["IKCS-2", "IKCS-1"]


def detailed_issue(reporter_id, comments=6):
    return NS(key="IKCS-7", fields=NS(
        summary="Laptop", description=None, status=NS(name="Open"), priority=NS(name="Low"), labels=[],
        created="2026-10-01T00:00", updated="2026-10-02T00:00", reporter=NS(accountId=reporter_id),
        comment=NS(comments=[NS(author=NS(displayName=f"agent{i}"), created="2026-10-0%dT" % (i + 1),
                                body=f"c{i}") for i in range(comments)]),
    ))


def test_get_ticket_details_returns_latest_comments(use_jira):
    use_jira(FakeJira(issue=detailed_issue("acc-asha")))
    details = jt.get_ticket_details("IKCS-7", EMAIL, max_comments=5)
    assert details["description"] == ""
    assert details["created"] == "2026-10-01"
    assert [c["body"] for c in details["comments"]] == ["c1", "c2", "c3", "c4", "c5"]


def test_get_ticket_details_hides_other_employees_tickets(use_jira):
    use_jira(FakeJira(issue=detailed_issue("someone-else")))
    with pytest.raises(ValueError, match="was not raised by"):
        jt.get_ticket_details("IKCS-7", EMAIL)
