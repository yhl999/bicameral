import pytest
from scripts.mcp_ingest_sessions import strip_untrusted_metadata, subchunk_evidence

def test_strip_untrusted_metadata():
    content = """
Hi there!

Conversation info (untrusted metadata):
```json
{
  "message_id": "123",
  "nested": "```python\\nfoo\\n```"
}
```

End of message.
"""
    expected = "Hi there!\n\nEnd of message."
    
    assert strip_untrusted_metadata(content) == expected

def test_strip_metadata_multiple_blocks():
    content = """
Block 1.

Sender (untrusted metadata):
```json
{"user": "Alice"}
```

Block 2.

Conversation info:
```json
{"id": "456"}
```

Block 3.
"""
    expected = "Block 1.\n\nBlock 2.\n\nBlock 3."
    assert strip_untrusted_metadata(content) == expected

def test_strip_metadata_no_match():
    content = "Hello\\n\\nWorld"
    assert strip_untrusted_metadata(content) == content.strip()

def test_strip_metadata_unclosed_fence():
    # An adversary doesn't close the fence. It shouldn't hang, and shouldn't strip.
    content = """
Conversation info (untrusted metadata):
```json
{"malicious": "payload"
"""
    # Without the closing fence, it shouldn't strip it.
    assert strip_untrusted_metadata(content) == content.strip()

def test_subchunk_evidence_uses_strip():
    content = """
Conversation info:
```json
{"id": "test"}
```
Test message.
"""
    # max_chars=100 won't split "Test message."
    chunks = subchunk_evidence(content.strip(), "my_key", 100)
    assert len(chunks) == 1
    assert chunks[0][0] == "my_key"
    assert chunks[0][1] == "Test message."
