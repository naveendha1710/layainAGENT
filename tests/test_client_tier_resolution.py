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
    assert call_kwargs["model"] == "meta/llama-3.1-8b-instruct"


def test_tier_big_resolves_to_big_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], tier="big")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "meta/llama-3.1-70b-instruct"


def test_model_literal_small_resolves_to_small_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], model="small")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "meta/llama-3.1-8b-instruct"


def test_model_literal_big_resolves_to_big_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], model="big")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "meta/llama-3.1-70b-instruct"


def test_explicit_custom_model_is_preserved() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}], model="custom/my-model:latest")
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "custom/my-model:latest"


def test_default_without_tier_or_model_resolves_to_big_model() -> None:
    client, mock_openai = _make_client_with_mock()
    client.chat_complete([{"role": "user", "content": "hi"}])
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "meta/llama-3.1-70b-instruct"
