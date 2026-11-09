import uuid
from types import SimpleNamespace

from langgraph.types import Command

from smartDeskAi import main
from smartDeskAi.main import TECHNICAL_ERROR, format_ticket_confirmation, new_session_config, run_turn


def test_new_session_config_tags_every_run_with_thread_id():
    config = new_session_config()
    thread_id = config["configurable"]["thread_id"]
    uuid.UUID(thread_id)
    assert config["metadata"] == {"thread_id": thread_id}
    assert config["tags"] == [f"thread_id:{thread_id}"]
    assert new_session_config()["configurable"]["thread_id"] != thread_id


def test_format_ticket_confirmation():
    text = format_ticket_confirmation({"summary": "VPN down", "description": "No connection", "priority": "High",
                                       "label": "IT", "email": "asha@synergy.com"})
    assert "Summary     : VPN down" in text and "Your email  : asha@synergy.com" in text
    assert text.rstrip().endswith("to add them to the description)")


class ScriptedGraph:
    def __init__(self, results):
        self.results = list(results)
        self.inputs = []

    def invoke(self, value, config):
        self.inputs.append(value)
        return self.results.pop(0)


def test_run_turn_resumes_interrupts_until_done(monkeypatch):
    interrupt = SimpleNamespace(id="int-1", value={"type": "ticket_confirmation", "ticket": {}})
    graph = ScriptedGraph([{"__interrupt__": [interrupt]}, {"final_response": "Ticket IKCS-1 created."}])
    monkeypatch.setattr(main, "ask_human", lambda value: "yes")

    assert run_turn(graph, "raise a ticket", {}) == "Ticket IKCS-1 created."
    assert isinstance(graph.inputs[1], Command) and graph.inputs[1].resume == {"int-1": "yes"}


def test_run_turn_without_final_response():
    assert run_turn(ScriptedGraph([{}]), "hi", {}) == TECHNICAL_ERROR


def test_ask_human_reprompts_until_answer(monkeypatch, capsys):
    answers = iter(["", "   ", "no"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert main.ask_human({"type": "other"}) == "no"
