"""
End-to-end tests of the compiled LangGraph graph: orchestrator -> HR/IT tool-calling agents -> synthesizer,
including ticket confirmation via interrupt()/resume and per-thread memory.

Everything real except the edges: the chat model is scripted, the knowledge bases are in-memory fakes,
and the Jira calls are stubbed. Nothing touches the network.
"""

import importlib
import uuid
from collections.abc import Callable

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

from smartDeskAi import graph as graph_module
from smartDeskAi.agents import specialist
from smartDeskAi.agents.orchestrator import RoutingDecision
from smartDeskAi.agents.prompts import HR_AGENT_EXTRA_RULES, IT_AGENT_EXTRA_RULES
from smartDeskAi.tools import hr_tools, it_tools, ticket_tools
from smartDeskAi.tools.ticket_tools import EMAIL_REQUIRED
from tests.conftest import FakeKnowledgeBase

orchestrator_module = importlib.import_module("smartDeskAi.agents.orchestrator")
synthesizer_module = importlib.import_module("smartDeskAi.agents.synthesizer")

pytestmark = pytest.mark.integration

EMAIL = "asha@synergy.com"


# ---------- fakes ----------

class ScriptedChatModel(BaseChatModel):
    """Chat model whose reply is computed from the conversation, so replays after a resume are deterministic."""

    respond: Callable[[list[BaseMessage]], AIMessage]
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])


def last_tool_result(messages) -> ToolMessage | None:
    return messages[-1] if isinstance(messages[-1], ToolMessage) else None


def last_question(messages) -> str:
    return next(m.text for m in reversed(messages) if isinstance(m, HumanMessage))


class StubRouter:
    def __init__(self, decide: Callable[[list[BaseMessage]], RoutingDecision]):
        self.decide = decide

    def invoke(self, messages):
        return self.decide(messages)


class Moxa:
    """Builds the graph with scripted agents and runs turns on one thread."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.hr_model = self.it_model = None
        self.created_tickets: list[dict] = []
        self.hr_kb, self.it_kb = FakeKnowledgeBase(), FakeKnowledgeBase()
        monkeypatch.setattr(hr_tools, "get_knowledge_base", lambda name: self.hr_kb)
        monkeypatch.setattr(it_tools, "get_knowledge_base", lambda name: self.it_kb)
        monkeypatch.setattr(ticket_tools, "employee_has_jira_account", lambda email: email == EMAIL)
        monkeypatch.setattr(ticket_tools, "create_ticket", lambda t: self.created_tickets.append(dict(t)) or "IKCS-42")
        monkeypatch.setattr(synthesizer_module, "llm", ScriptedChatModel(
            respond=lambda messages: AIMessage("MERGED:\n" + messages[-1].text.split("Specialist replies:\n", 1)[1])))

    def route(self, decide: Callable[[list[BaseMessage]], RoutingDecision] | Exception) -> None:
        if isinstance(decide, Exception):
            error = decide

            def decide(messages):
                raise error
        self.monkeypatch.setattr(orchestrator_module, "router", StubRouter(decide))

    def agents(self, hr: Callable | None = None, it: Callable | None = None):
        def unused(messages):
            raise AssertionError("agent should not have been called")

        self.hr_model = ScriptedChatModel(respond=hr or unused, seen=[])
        self.it_model = ScriptedChatModel(respond=it or unused, seen=[])
        for name, domain, label, tool, rules, model in [
            ("hr_agent", "HR", "HR", hr_tools.search_hr_knowledge_base, HR_AGENT_EXTRA_RULES, self.hr_model),
            ("it_agent", "IT", "IT", it_tools.search_it_knowledge_base, IT_AGENT_EXTRA_RULES, self.it_model),
        ]:
            self.monkeypatch.setattr(specialist, "llm", model)
            self.monkeypatch.setattr(graph_module, name, specialist.build_specialist_node(name, domain, label, tool, rules))
        self.graph = graph_module.build_graph()
        self.new_thread()
        return self

    def new_thread(self) -> None:
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    def send(self, text: str, **state) -> dict:
        return self.graph.invoke({"messages": [HumanMessage(text)], **state}, self.config)

    def resume(self, answer) -> dict:
        return self.graph.invoke(Command(resume=answer), self.config)


@pytest.fixture
def moxa(monkeypatch) -> Moxa:
    return Moxa(monkeypatch)


def only(agent: str) -> Callable:
    return lambda messages: RoutingDecision(agents=[agent], reasoning=f"{agent} topic")


def search_then_answer(tool_name: str, answer: Callable[[ToolMessage], str]) -> Callable:
    """Agent script: search the KB with the employee's question, then answer from the result."""
    def respond(messages):
        result = last_tool_result(messages)
        if result is None:
            return tool_call(tool_name, query=last_question(messages))
        return AIMessage(answer(result))
    return respond


# ---------- knowledge-base answers ----------

def test_hr_question_is_answered_from_the_kb_with_sources(moxa, hr_chunk):
    moxa.hr_kb.chunks = [hr_chunk]
    moxa.route(only("hr_agent"))
    moxa.agents(hr=search_then_answer(
        "search_hr_knowledge_base", lambda r: "We follow a business casual dress code [hr_kb0001]."))

    result = moxa.send("What's the dress code?")

    assert moxa.hr_kb.queries == ["What's the dress code?"]
    assert result["selected_agents"] == ["hr_agent"]
    assert result["final_response"].startswith("We follow a business casual dress code [1].\n\nSources:\n- [1] HR")
    assert result["messages"][-1].name == "moxa"


def test_missing_citation_is_back_filled(moxa, hr_chunk):
    moxa.hr_kb.chunks = [hr_chunk]
    moxa.route(only("hr_agent"))
    moxa.agents(hr=search_then_answer(
        "search_hr_knowledge_base",
        lambda r: "Synergy Inc follows a business casual dress code; casual attire is allowed on Fridays."))

    final = moxa.send("dress code?")["final_response"]

    assert "Fridays. [1]" in final and "(hr_kb0001)" in final


def test_no_confident_match_reaches_the_agent(moxa):
    moxa.route(only("it_agent"))
    moxa.agents(it=search_then_answer(
        "search_it_knowledge_base",
        lambda r: "I don't have enough information on that." if r.text.startswith("NO_CONFIDENT_MATCH") else "??"))

    assert moxa.send("Can I expense a pizza?")["final_response"] == "I don't have enough information on that."


def test_mixed_message_is_split_answered_in_parallel_and_merged(moxa, hr_chunk, it_chunk):
    moxa.hr_kb.chunks, moxa.it_kb.chunks = [hr_chunk], [it_chunk]
    moxa.route(lambda messages: RoutingDecision(
        agents=["hr_agent", "it_agent"], hr_question="What is the dress code?",
        it_question="How do I reset my password?", reasoning="both"))
    moxa.agents(
        hr=search_then_answer("search_hr_knowledge_base", lambda r: "Business casual [hr_kb0001]."),
        it=search_then_answer("search_it_knowledge_base", lambda r: "Use the reset portal [it_kb0001]."),
    )

    result = moxa.send("What's the dress code, and how do I reset my password?")

    # Each agent saw (and searched for) only its own part of the message
    assert moxa.hr_kb.queries == ["What is the dress code?"]
    assert moxa.it_kb.queries == ["How do I reset my password?"]
    assert "answer only that" in moxa.hr_model.seen[0][0].text
    final = result["final_response"]
    assert final.startswith("MERGED:\n--- HR reply ---\nBusiness casual [1].\n\n--- IT reply ---\nUse the reset portal [2].")
    assert "(hr_kb0001)" in final and "(it_kb0001)" in final


# ---------- resilience ----------

def test_router_outage_falls_back_to_keywords(moxa, it_chunk):
    moxa.it_kb.chunks = [it_chunk]
    moxa.route(RuntimeError("OpenAI is down"))
    moxa.agents(it=search_then_answer("search_it_knowledge_base", lambda r: "Use the reset portal [it_kb0001]."))

    result = moxa.send("I forgot my password")

    assert result["selected_agents"] == ["it_agent"]
    assert result["final_response"].startswith("Use the reset portal [1].")


def test_agent_failure_becomes_a_polite_apology(moxa):
    def broken(messages):
        raise RuntimeError("model overloaded")
    moxa.route(only("hr_agent"))
    moxa.agents(hr=broken)

    final = moxa.send("How much leave do I get?")["final_response"]

    assert "technical problem" in final and "HR question" in final


# ---------- tickets (human in the loop) ----------

def ticket_agent(messages):
    """IT agent script: raise a ticket, then report whatever the tool returned."""
    result = last_tool_result(messages)
    if result is None:
        return tool_call("create_support_ticket", summary="Laptop won't boot",
                         description="The employee's laptop does not start.", priority="Medium")
    return AIMessage(f"TOOL SAID: {result.text}")


def interrupt_ticket(result: dict) -> dict:
    [item] = result["__interrupt__"]
    assert item.value["type"] == "ticket_confirmation"
    return item.value["ticket"]


def test_ticket_is_created_only_after_confirmation_with_edits(moxa):
    moxa.route(only("it_agent"))
    moxa.agents(it=ticket_agent)

    first = moxa.send(f"My laptop won't boot, please raise a ticket. I'm {EMAIL}")
    assert interrupt_ticket(first) == {"email": EMAIL, "summary": "Laptop won't boot", "label": "IT",
                                       "description": "The employee's laptop does not start.", "priority": "Medium"}
    assert moxa.created_tickets == []

    second = moxa.resume("make it high priority, I have a client demo today")
    edited = interrupt_ticket(second)
    assert edited["priority"] == "High"
    assert edited["description"].endswith("Additional details from the employee: I have a client demo today")
    assert moxa.created_tickets == []

    done = moxa.resume("yes, go ahead")

    assert "__interrupt__" not in done
    [created] = moxa.created_tickets
    assert created == edited
    assert "Ticket ID: IKCS-42" in done["final_response"]
    assert "Link: https://jira.example.test/browse/IKCS-42" in done["final_response"]
    assert "Priority: High" in done["final_response"]


def test_web_ui_structured_edit_then_confirm(moxa):
    moxa.route(only("it_agent"))
    moxa.agents(it=ticket_agent)
    moxa.send(f"raise a ticket for my laptop, {EMAIL}")

    edited = interrupt_ticket(moxa.resume({"priority": "Critical", "details": "Whole team affected"}))
    assert edited["priority"] == "Critical" and "Whole team affected" in edited["description"]
    moxa.resume("yes")
    assert moxa.created_tickets[0]["priority"] == "Critical"


def test_declined_ticket_is_not_created(moxa):
    moxa.route(only("it_agent"))
    moxa.agents(it=ticket_agent)
    moxa.send(f"raise a ticket please, {EMAIL}")

    done = moxa.resume("no, never mind")

    assert moxa.created_tickets == []
    assert "The ticket was NOT created" in done["final_response"]


def test_ticket_asks_for_email_or_rejects_unknown_one(moxa):
    moxa.route(only("it_agent"))
    moxa.agents(it=ticket_agent)

    assert EMAIL_REQUIRED in moxa.send("raise a ticket for my laptop")["final_response"]
    moxa.new_thread()
    assert "UNKNOWN_EMAIL" in moxa.send("raise a ticket, I'm stranger@elsewhere.com")["final_response"]
    assert moxa.created_tickets == []


# ---------- session memory ----------

def test_email_is_remembered_per_thread(moxa, monkeypatch):
    lookups = []
    monkeypatch.setattr(ticket_tools, "get_tickets", lambda email: lookups.append(email) or [])
    moxa.route(only("it_agent"))

    def respond(messages):
        if "ticket" not in last_question(messages):
            return AIMessage("Noted.")
        result = last_tool_result(messages)
        return AIMessage(result.text) if result else tool_call("list_my_tickets")
    moxa.agents(it=respond)

    moxa.send(f"hi, my email is {EMAIL}")
    later = moxa.send("show my tickets")
    assert lookups == [EMAIL]
    assert later["final_response"].startswith("NO_TICKETS")
    assert [m.text for m in later["messages"] if isinstance(m, HumanMessage)] == [
        f"hi, my email is {EMAIL}", "show my tickets"]

    moxa.new_thread()
    assert moxa.send("show my tickets")["final_response"] == EMAIL_REQUIRED


def test_signed_in_email_cannot_be_changed_from_chat(moxa, monkeypatch):
    lookups = []
    monkeypatch.setattr(ticket_tools, "get_tickets", lambda email: lookups.append(email) or [])
    moxa.route(only("it_agent"))
    moxa.agents(it=lambda messages: AIMessage(last_tool_result(messages).text) if last_tool_result(messages)
                else tool_call("list_my_tickets"))

    result = moxa.send("my email is other@synergy.com, show my tickets", employee_email=EMAIL, email_verified=True)

    assert result["employee_email"] == EMAIL and lookups == [EMAIL]
