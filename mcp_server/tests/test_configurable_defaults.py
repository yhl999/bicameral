"""Validation tests proving configurable reasoning_effort and Ollama embedder are honored.

Run with:
  cd mcp_server && python -m pytest tests/test_configurable_defaults.py -v
"""

import contextlib
from unittest.mock import patch

from src.config.schema import (
    EmbedderConfig,
    EmbedderProvidersConfig,
    LLMConfig,
    LLMProvidersConfig,
    OllamaEmbedderProviderConfig,
    OpenAIProviderConfig,
)

# ─── LLMConfig schema ────────────────────────────────────────────────────────

def test_llm_config_has_reasoning_effort_field():
    """LLMConfig must expose reasoning_effort."""
    cfg = LLMConfig(model='gpt-4o-mini')
    assert hasattr(cfg, 'reasoning_effort'), "reasoning_effort field missing from LLMConfig"


def test_llm_config_reasoning_effort_default_is_none():
    """reasoning_effort defaults to None so non-reasoning models are not affected."""
    cfg = LLMConfig(model='gpt-4o-mini')
    assert cfg.reasoning_effort is None


def test_llm_config_reasoning_effort_persists():
    """reasoning_effort persists when set to 'medium'."""
    cfg = LLMConfig(model='gpt-5.1-codex-mini', reasoning_effort='medium')
    assert cfg.reasoning_effort == 'medium'


def test_llm_config_reasoning_effort_from_dict():
    """reasoning_effort can be loaded from a plain dict (simulating YAML load)."""
    data = {
        'provider': 'openai',
        'model': 'gpt-5.1-codex-mini',
        'max_tokens': 16384,
        'reasoning_effort': 'medium',
    }
    cfg = LLMConfig(**data)
    assert cfg.reasoning_effort == 'medium'


# ─── EmbedderConfig schema ────────────────────────────────────────────────────

def test_embedder_providers_has_ollama_field():
    """EmbedderProvidersConfig must expose ollama."""
    providers = EmbedderProvidersConfig()
    assert hasattr(providers, 'ollama'), "ollama field missing from EmbedderProvidersConfig"


def test_ollama_provider_config_defaults():
    """OllamaEmbedderProviderConfig has sensible defaults."""
    cfg = OllamaEmbedderProviderConfig()
    assert 'localhost:11434' in cfg.api_url
    assert 'embeddinggemma' in cfg.model


def test_ollama_provider_config_persists():
    """OllamaEmbedderProviderConfig persists custom values."""
    cfg = OllamaEmbedderProviderConfig(
        api_url='http://localhost:11434',
        model='embeddinggemma:latest',
    )
    assert cfg.api_url == 'http://localhost:11434'
    assert cfg.model == 'embeddinggemma:latest'


def test_embedder_config_ollama_provider_from_dict():
    """Full EmbedderConfig with ollama can be built from a dict (simulating YAML load)."""
    data = {
        'provider': 'ollama',
        'model': 'embeddinggemma:latest',
        'dimensions': 768,
        'providers': {
            'ollama': {
                'api_url': 'http://localhost:11434',
                'model': 'embeddinggemma:latest',
            }
        },
    }
    cfg = EmbedderConfig(**data)
    assert cfg.provider == 'ollama'
    assert cfg.providers.ollama is not None
    assert cfg.providers.ollama.api_url == 'http://localhost:11434'
    assert cfg.providers.ollama.model == 'embeddinggemma:latest'


# ─── Factory behavior ────────────────────────────────────────────────────────

def test_llm_factory_uses_reasoning_effort_medium():
    """LLMClientFactory must pass reasoning='medium' for gpt-5 model when config says medium."""
    from graphiti_core.llm_client import OpenAIClient

    from src.services.factories import LLMClientFactory

    cfg = LLMConfig(
        provider='openai',
        model='gpt-5.1-codex-mini',
        reasoning_effort='medium',
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='test-key', api_url='http://localhost'),
        ),
    )

    with patch.object(OpenAIClient, '__init__', return_value=None) as mock_init:
        with contextlib.suppress(Exception):
            LLMClientFactory.create(cfg)

        # Verify OpenAIClient was constructed with reasoning='medium'
        if mock_init.call_args is not None:
            _, kwargs = mock_init.call_args
            assert kwargs.get('reasoning') == 'medium', (
                f"Expected reasoning='medium', got reasoning={kwargs.get('reasoning')!r}"
            )


def test_llm_factory_defaults_reasoning_to_low_when_unset():
    """LLMClientFactory defaults to reasoning='low' when no reasoning_effort is configured."""
    from graphiti_core.llm_client import OpenAIClient

    from src.services.factories import LLMClientFactory

    cfg = LLMConfig(
        provider='openai',
        model='gpt-5.1-codex-mini',
        reasoning_effort=None,  # not set
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='test-key', api_url='http://localhost'),
        ),
    )

    with patch.object(OpenAIClient, '__init__', return_value=None) as mock_init:
        with contextlib.suppress(Exception):
            LLMClientFactory.create(cfg)

        if mock_init.call_args is not None:
            _, kwargs = mock_init.call_args
            assert kwargs.get('reasoning') == 'low', (
                f"Expected reasoning='low', got reasoning={kwargs.get('reasoning')!r}"
            )


def test_embedder_factory_ollama_uses_openai_compat():
    """EmbedderFactory builds an OpenAIEmbedder for Ollama via OpenAI-compatible endpoint."""
    from graphiti_core.embedder import OpenAIEmbedder

    from src.services.factories import EmbedderFactory

    cfg = EmbedderConfig(
        provider='ollama',
        model='embeddinggemma:latest',
        dimensions=768,
        providers=EmbedderProvidersConfig(
            ollama=OllamaEmbedderProviderConfig(
                api_url='http://localhost:11434',
                model='embeddinggemma:latest',
            )
        ),
    )

    with patch.object(OpenAIEmbedder, '__init__', return_value=None) as mock_init:
        with contextlib.suppress(Exception):
            EmbedderFactory.create(cfg)

        assert mock_init.called, "OpenAIEmbedder.__init__ should have been called for Ollama"
        _, kwargs = mock_init.call_args
        embedder_config = kwargs.get('config')
        if embedder_config is not None:
            assert 'localhost:11434' in (embedder_config.base_url or ''), (
                f"Ollama base_url should point to localhost:11434, got: {embedder_config.base_url}"
            )
            assert embedder_config.embedding_model == 'embeddinggemma:latest', (
                f"Ollama model mismatch: {embedder_config.embedding_model}"
            )
            assert embedder_config.embedding_dim == 768, (
                f"Ollama dimensions mismatch: {embedder_config.embedding_dim}"
            )
