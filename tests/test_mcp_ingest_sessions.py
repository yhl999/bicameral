import pytest

from scripts.mcp_ingest_sessions import (
    MCPClient,
    _validate_mcp_url,
    strip_untrusted_metadata,
    subchunk_evidence,
)


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


# ---------------------------------------------------------------------------
# NEW: SSRF hardening — _validate_mcp_url
# ---------------------------------------------------------------------------


def test_validate_mcp_url_allows_localhost() -> None:
    """Localhost MCP URL is explicitly allowed (expected deployment target)."""
    assert _validate_mcp_url("http://localhost:8000/mcp") == "http://localhost:8000/mcp"


def test_validate_mcp_url_allows_private_ip() -> None:
    """Private RFC-1918 IPs are allowed (MCP server may run on local network)."""
    assert _validate_mcp_url("http://192.168.1.50:8000/mcp") == "http://192.168.1.50:8000/mcp"


def test_validate_mcp_url_allows_https() -> None:
    """HTTPS is always valid for MCP."""
    assert _validate_mcp_url("https://graphiti.internal/mcp") == "https://graphiti.internal/mcp"


def test_validate_mcp_url_rejects_non_http_scheme() -> None:
    """Non-http(s) schemes (file://, gopher://) must be rejected."""
    with pytest.raises(ValueError, match="http"):
        _validate_mcp_url("file:///etc/passwd")


def test_validate_mcp_url_rejects_gopher() -> None:
    with pytest.raises(ValueError, match="http"):
        _validate_mcp_url("gopher://localhost/evil")


def test_validate_mcp_url_rejects_embedded_credentials() -> None:
    """URLs with embedded user:pass credentials are rejected."""
    with pytest.raises(ValueError, match="credentials"):
        _validate_mcp_url("http://user:secret@localhost:8000/mcp")


def test_validate_mcp_url_rejects_link_local_cloud_metadata() -> None:
    """Cloud metadata IP 169.254.169.254 (link-local) is always blocked."""
    with pytest.raises(ValueError, match="link-local"):
        _validate_mcp_url("http://169.254.169.254/latest/meta-data/")


def test_validate_mcp_url_rejects_empty_netloc() -> None:
    """URL with no host is rejected."""
    with pytest.raises(ValueError, match="http"):
        _validate_mcp_url("http:///mcp")


def test_mcp_client_validates_url_on_init() -> None:
    """MCPClient raises ValueError on init when URL fails validation."""
    with pytest.raises(ValueError):
        MCPClient("file:///etc/passwd")


def test_mcp_client_accepts_localhost_url() -> None:
    """MCPClient accepts localhost MCP URL on init."""
    client = MCPClient("http://localhost:8000/mcp")
    assert client.url == "http://localhost:8000/mcp"
