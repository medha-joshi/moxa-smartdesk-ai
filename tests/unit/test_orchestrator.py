import importlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from smartDeskAi.agents.orchestrator import (
    AGENT_HISTORY, ROUTING_HISTORY, RoutingDecision, extract_email, fallback_route, resolve_employee_email,
    routing_input,
)

# smartDeskAi.agents re-exports the node function under the module's name, so fetch the module itself
orch = importlib.import_module("smartDeskAi.agents.orchestrator")


class StubRouter:
    def __init__(self, decision: RoutingDecision | Exception):
        self.decision = decision
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


@pytest.fixture
def route_to(monkeypatch):
    def install(decision):
        router = StubRouter(decision)
        monkeypatch.setattr(orch, "router", router)
        return router
    return install


# ---------- email handling ----------

@pytest.mark.parametrize("text, expected", [
    ("my email is Asha.Rao@Synergy.com thanks", "asha.rao@synergy.com"),
    ("reach me at a.b+tag@mail.co.uk.", "a.b+tag@mail.co.uk"),
    ("no address here", None),
    ("", None),
    (None, None),
])
def test_extract_email(text, expected):
    assert extract_email(text) == expected


def test_first_email_is_remembered():
    assert resolve_employee_email("I'm asha@synergy.com", None) == "asha@synergy.com"


def test_no_email_in_message_keeps_known():
    assert resolve_employee_email("my vpn is broken", "asha@synergy.com") == "asha@synergy.com"


@pytest.mark.parametrize("text", [
    "sorry, my email is ravi@synergy.com",
    "my correct email address is ravi@synergy.com",
    "use ravi@synergy.com",
    "ravi@synergy.com",
    "ravi@synergy.com.",
])
def test_explicit_self_email_replaces_known(text):
    assert resolve_employee_email(text, "asha@synergy.com") == "ravi@synergy.com"


@pytest.mark.parametrize("text", [
    "can you cc my manager ravi@synergy.com",
    "ravi@synergy.com also has this problem",
])
def test_colleague_email_does_not_replace_known(text):
    assert resolve_employee_email(text, "asha@synergy.com") == "asha@synergy.com"


# ---------- keyword fallback ----------

@pytest.mark.parametrize("text, agents", [
    ("How many sick days do I get?", ["hr_agent"]),
    ("My VPN keeps dropping", ["it_agent"]),
    ("Reset my password and tell me about maternity leave", ["hr_agent", "it_agent"]),
    ("hello there", ["hr_agent"]),  # default
    ("it is raining", ["hr_agent"]),  # lowercase "it" is not IT
    ("IT please help", ["it_agent"]),
])
def test_fallback_route(text, agents):
    assert fallback_route(text).agents == agents


def test_routing_input_trims_history():
    messages = [HumanMessage(f"m{i}") for i in range(ROUTING_HISTORY + 5)]
    result = routing_input(messages)
    assert isinstance(result[0], SystemMessage)
    assert [m.text for m in result[1:]] == [m.text for m in messages[-ROUTING_HISTORY:]]


# ---------- orchestrator node ----------

def test_orchestrator_routes_to_single_agent(route_to):
    route_to(RoutingDecision(agents=["it_agent"], reasoning="VPN issue"))
    state = {"messages": [HumanMessage("VPN down, I'm asha@synergy.com")]}

    command = orch.orchestrator(state)

    assert command.update["selected_agents"] == ["it_agent"]
    assert command.update["employee_email"] == "asha@synergy.com"
    assert command.update["agent_responses"] is None  # clears the last turn
    assert command.update["routing_reasoning"] == "VPN issue"
    [send] = command.goto
    assert isinstance(send, Send) and send.node == "it_agent"
    assert send.arg["question"] is None
    assert send.arg["employee_email"] == "asha@synergy.com"


def test_orchestrator_splits_mixed_message(route_to):
    route_to(RoutingDecision(agents=["hr_agent", "it_agent", "hr_agent"], hr_question=" Leave policy? ",
                             it_question="Reset password?", reasoning="both"))
    command = orch.orchestrator({"messages": [HumanMessage("leave policy and password reset")]})

    assert command.update["selected_agents"] == ["hr_agent", "it_agent"]  # de-duplicated, order kept
    questions = {send.node: send.arg["question"] for send in command.goto}
    assert questions == {"hr_agent": "Leave policy?", "it_agent": "Reset password?"}


def test_orchestrator_does_not_split_when_a_part_is_missing(route_to):
    route_to(RoutingDecision(agents=["hr_agent", "it_agent"], hr_question="Leave?", it_question="", reasoning="x"))
    command = orch.orchestrator({"messages": [HumanMessage("leave and laptop")]})
    assert all(send.arg["question"] is None for send in command.goto)


def test_orchestrator_defaults_to_hr_when_router_returns_nothing(route_to):
    route_to(RoutingDecision(agents=[], reasoning="unclear"))
    command = orch.orchestrator({"messages": [HumanMessage("hmm")]})
    assert [send.node for send in command.goto] == ["hr_agent"]


def test_orchestrator_uses_keyword_fallback_when_router_fails(route_to):
    route_to(RuntimeError("OpenAI down"))
    command = orch.orchestrator({"messages": [HumanMessage("my laptop won't boot")]})
    assert command.update["selected_agents"] == ["it_agent"]
    assert "fallback" in command.update["routing_reasoning"].lower()


def test_verified_email_is_never_replaced_from_chat(route_to):
    route_to(RoutingDecision(agents=["hr_agent"], reasoning="x"))
    state = {"messages": [HumanMessage("my email is someone.else@synergy.com")],
             "employee_email": "asha@synergy.com", "email_verified": True}
    assert orch.orchestrator(state).update["employee_email"] == "asha@synergy.com"


def test_orchestrator_uses_latest_human_message_and_limits_agent_history(route_to):
    route_to(RoutingDecision(agents=["hr_agent"], reasoning="x"))
    history = [HumanMessage(f"q{i}") if i % 2 == 0 else AIMessage(f"a{i}") for i in range(AGENT_HISTORY + 6)]
    history.append(HumanMessage("final: email me at new@synergy.com"))
    history.append(AIMessage("trailing AI message"))

    command = orch.orchestrator({"messages": history})

    assert command.update["employee_email"] == "new@synergy.com"
    assert len(command.goto[0].arg["messages"]) == AGENT_HISTORY
