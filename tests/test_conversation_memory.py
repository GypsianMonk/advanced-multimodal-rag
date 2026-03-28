"""tests/test_conversation_memory.py — Multi-turn conversation memory tests."""

import pytest
from memory.conversation_memory import ConversationMemory, Conversation


@pytest.fixture
def mem():
    return ConversationMemory(max_conversations=10)


def test_create_conversation(mem):
    conv = mem.create()
    assert conv.conversation_id
    assert mem.get(conv.conversation_id) is conv


def test_add_turns(mem):
    conv = mem.create()
    mem.add_user_turn(conv.conversation_id, "What is fraud?")
    mem.add_assistant_turn(conv.conversation_id, "Fraud is...", confidence=0.9)
    fetched = mem.get(conv.conversation_id)
    assert len(fetched.turns) == 2
    assert fetched.turns[0].role == "user"
    assert fetched.turns[1].role == "assistant"


def test_last_user_query(mem):
    conv = mem.create()
    mem.add_user_turn(conv.conversation_id, "First question")
    mem.add_assistant_turn(conv.conversation_id, "Answer")
    mem.add_user_turn(conv.conversation_id, "Follow-up question")
    assert mem.get(conv.conversation_id).last_user_query() == "Follow-up question"


def test_build_contextual_query(mem):
    conv = mem.create()
    mem.add_user_turn(conv.conversation_id, "What was Q3 revenue?")
    mem.add_assistant_turn(conv.conversation_id, "Revenue was $48.2M")
    enriched = mem.build_contextual_query(conv.conversation_id, "What about Q2?")
    assert "Q3 revenue" in enriched
    assert "What about Q2?" in enriched


def test_get_or_create_new(mem):
    conv = mem.get_or_create(None)
    assert conv.conversation_id in [c["conversation_id"] for c in mem.list_conversations()]


def test_get_or_create_existing(mem):
    conv = mem.create()
    conv2 = mem.get_or_create(conv.conversation_id)
    assert conv.conversation_id == conv2.conversation_id


def test_list_conversations(mem):
    for _ in range(3):
        mem.create()
    listings = mem.list_conversations()
    assert len(listings) == 3
    assert all("conversation_id" in c for c in listings)


def test_max_conversations_eviction(mem):
    for _ in range(15):
        mem.create()
    assert len(mem.list_conversations()) <= 10


def test_dialogue_string(mem):
    conv = mem.create()
    mem.add_user_turn(conv.conversation_id, "Hello")
    mem.add_assistant_turn(conv.conversation_id, "Hi there")
    c = mem.get(conv.conversation_id)
    dialogue = c.to_dialogue_string()
    assert "User: Hello" in dialogue
    assert "Assistant: Hi there" in dialogue
