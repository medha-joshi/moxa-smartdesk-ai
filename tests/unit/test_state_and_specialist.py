from langchain_core.messages import AIMessage, HumanMessage

from smartDeskAi.agents.specialist import own_part
from smartDeskAi.state import merge_agent_responses

HR = {"agent": "hr_agent", "response": "hr"}
IT = {"agent": "it_agent", "response": "it"}


def test_merge_agent_responses_accumulates_parallel_updates():
    assert merge_agent_responses(None, [HR]) == [HR]
    assert merge_agent_responses([HR], [IT]) == [HR, IT]


def test_merge_agent_responses_none_clears_for_new_turn():
    assert merge_agent_responses([HR, IT], None) == []


def test_own_part_without_question_returns_copy():
    messages = [HumanMessage("hi")]
    result = own_part(messages, None)
    assert result == messages and result is not messages


def test_own_part_replaces_only_latest_human_message():
    messages = [HumanMessage("first"), AIMessage("reply"), HumanMessage("leave and vpn?"), AIMessage("tail")]
    result = own_part(messages, "What is the leave policy?")
    assert [m.text for m in result] == ["first", "reply", "What is the leave policy?", "tail"]
    assert isinstance(result[2], HumanMessage)


def test_own_part_with_no_human_message():
    messages = [AIMessage("only ai")]
    assert own_part(messages, "question") == messages
