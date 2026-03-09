#!/usr/bin/env python3
"""Regression tests for LLM factory max_tokens wiring."""

from pathlib import Path
import sys

# Add the mcp_server src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.schema import LLMConfig, LLMProvidersConfig, OpenAIProviderConfig
import services.factories as factories


def test_openai_factory_passes_configured_max_tokens(monkeypatch):
    captured: dict[str, object] = {}

    class DummyOpenAIClient:
        def __init__(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs

    monkeypatch.setattr(factories, 'OpenAIClient', DummyOpenAIClient)

    config = LLMConfig(
        provider='openai',
        model='gpt-4o-mini',
        max_tokens=4096,
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='dummy-key', api_url='https://example.invalid/v1')
        ),
    )

    factories.LLMClientFactory.create(config)

    assert captured['kwargs']['max_tokens'] == 4096
    llm_config = captured['kwargs']['config']
    assert llm_config.max_tokens == 4096


def test_reasoning_factory_passes_configured_max_tokens(monkeypatch):
    captured: dict[str, object] = {}

    class DummyOpenAIClient:
        def __init__(self, *args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs

    monkeypatch.setattr(factories, 'OpenAIClient', DummyOpenAIClient)

    config = LLMConfig(
        provider='openai',
        model='gpt-5-mini',
        max_tokens=2048,
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='dummy-key', api_url='https://example.invalid/v1')
        ),
    )

    factories.LLMClientFactory.create(config)

    assert captured['kwargs']['max_tokens'] == 2048
    assert captured['kwargs']['reasoning'] == 'minimal'
    assert captured['kwargs']['verbosity'] == 'low'
