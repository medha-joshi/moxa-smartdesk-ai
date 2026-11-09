import importlib

from langchain_core.messages import AIMessage, HumanMessage

from smartDeskAi.agents.synthesizer import NO_RESPONSE, synthesizer

# smartDeskAi.agents re-exports the node function under the module's name, so fetch the module itself
synth = importlib.import_module("smartDeskAi.agents.synthesizer")


class StubLLM:
    def __init__(self, reply: str | Exception):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return AIMessage(self.reply)


STATE = {"messages": [HumanMessage("dress code and password reset?")]}
HR = {"agent": "hr_agent", "response": "Business casual [hr_kb0001]."}
IT = {"agent": "it_agent", "response": "Use the reset portal [it_kb0001]."}


def test_no_responses():
    result = synthesizer({**STATE, "agent_responses": []})
    assert result["final_response"] == NO_RESPONSE


def test_single_response_passes_through_without_llm(monkeypatch):
    llm = StubLLM(AssertionError("should not be called"))
    monkeypatch.setattr(synth, "llm", llm)
    result = synthesizer({**STATE, "agent_responses": [HR]})
    assert result["final_response"].startswith("Business casual [1].\n\nSources:")
    assert result["messages"][0].name == "moxa"
    assert llm.prompts == []


def test_two_responses_are_merged_hr_first(monkeypatch):
    llm = StubLLM("Merged answer [hr_kb0001] [it_kb0001]")
    monkeypatch.setattr(synth, "llm", llm)

    result = synthesizer({**STATE, "agent_responses": [IT, HR]})

    prompt = llm.prompts[0][1].text
    assert prompt.index("--- HR reply ---") < prompt.index("--- IT reply ---")
    assert "dress code and password reset?" in prompt
    assert result["final_response"].startswith("Merged answer [1] [2]")


def test_llm_failure_falls_back_to_concatenation(monkeypatch):
    monkeypatch.setattr(synth, "llm", StubLLM(RuntimeError("timeout")))
    final = synthesizer({**STATE, "agent_responses": [HR, IT]})["final_response"]
    assert final.startswith("HR:\nBusiness casual [1].\n\nIT:\nUse the reset portal [2].")


def test_citation_failure_keeps_answer(monkeypatch):
    def broken(text):
        raise OSError("kb file missing")
    monkeypatch.setattr(synth, "add_citations", broken)
    assert synthesizer({**STATE, "agent_responses": [HR]})["final_response"] == HR["response"]
