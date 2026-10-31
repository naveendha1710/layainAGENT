"""Unit tests for tier-to-model resolution in NVIDIABuildClient."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from laya_agent.agent.llm.client import NVIDIABuildClient


def _make_client_with_mock() -> tuple[NVIDIABuildClient, MagicMock]:
    client = NVIDIABuildClient(api_key="mock-key")
    mock_openai = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.model = "resolved-model"
    mock_openai.chat.completions.create.return_value = mock_response
    client._openai = mock_openai
    return client, mock_openai


def test_tier_small_resolves_to_small_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], tier="small")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "nvidia/nemotron-3.5-bolt-e4b"


def test_tier_big_resolves_to_big_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], tier="big")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_model_literal_small_resolves_to_small_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], model="small")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "nvidia/nemotron-3.5-bolt-e4b"


def test_model_literal_big_resolves_to_big_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], model="big")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_unmapped_model_raises_value_error() -> None:
    client, _ = _make_client_with_mock()
    with pytest.raises(ValueError, match="Unspecified or invalid model parameter"):
        client.chat_complete([{"role": "user", "content": "hi"}], model="custom/unmapped-model")


def test_missing_model_raises_value_error() -> None:
    client, _ = _make_client_with_mock()
    with pytest.raises(ValueError, match="Unspecified or invalid model parameter"):
        client.chat_complete([{"role": "user", "content": "hi"}])


def test_missing_tier_config_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BIG_TIER_MODEL", raising=False)
    monkeypatch.delenv("SMALL_TIER_MODEL", raising=False)
    monkeypatch.setenv("LAYA_CONFIG_PATH", "non_existent_config.yaml")
    with pytest.raises(ValueError, match="Research evaluation requires explicit BIG_TIER_MODEL and SMALL_TIER_MODEL"):
        NVIDIABuildClient(api_key="mock-key", big_tier_model=None, small_tier_model=None)


def test_max_tokens_passed_to_openai_client() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], model="big", max_tokens=512)
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 512
