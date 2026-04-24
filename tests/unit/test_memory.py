"""Unit tests for memory systems."""

from effgen.memory.short_term import Message, MessageRole, ShortTermMemory


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.timestamp > 0

    def test_message_to_dict(self):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
        assert "timestamp" in d

    def test_message_from_dict(self):
        d = {"role": "user", "content": "Test message"}
        msg = Message.from_dict(d)
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"

    def test_message_estimate_tokens(self):
        msg = Message(role=MessageRole.USER, content="Hello world test message here")
        tokens = msg.estimate_tokens()
        assert tokens > 0
        assert tokens < 100

    def test_message_roles(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    def test_create_empty(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        assert len(mem.messages) == 0

    def test_add_message(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        assert len(mem.messages) == 1

    def test_add_multiple_messages(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        mem.add_message(MessageRole.ASSISTANT, "Hi!")
        mem.add_message(MessageRole.USER, "How are you?")
        assert len(mem.messages) == 3

    def test_get_recent_messages(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        mem.add_message(MessageRole.ASSISTANT, "Hi!")
        messages = mem.get_recent_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi!"

    def test_get_recent_messages_limited(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "A")
        mem.add_message(MessageRole.ASSISTANT, "B")
        mem.add_message(MessageRole.USER, "C")
        messages = mem.get_recent_messages(n=2)
        assert len(messages) == 2
        assert messages[0].content == "B"
        assert messages[1].content == "C"

    def test_clear(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "Hello")
        mem.add_message(MessageRole.ASSISTANT, "Hi!")
        mem.clear()
        assert len(mem.messages) == 0

    def test_total_messages_counter(self):
        mem = ShortTermMemory(max_tokens=4096, max_messages=100)
        mem.add_message(MessageRole.USER, "A")
        mem.add_message(MessageRole.USER, "B")
        assert mem.total_messages_added == 2
