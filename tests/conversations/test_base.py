"""Tests for conversation-history repository contracts."""

import inspect

import pytest

from tuesday.conversations import (
    BaseConversationRepository,
    ConversationRepositoryError,
)


def test_repository_error_is_runtime_error() -> None:
    assert issubclass(ConversationRepositoryError, RuntimeError)


def test_base_repository_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseConversationRepository()


def test_base_repository_declares_exact_abstract_operations() -> None:
    assert BaseConversationRepository.__abstractmethods__ == {
        "append_messages",
        "get_context",
    }


def test_repository_operations_are_asynchronous() -> None:
    assert inspect.iscoroutinefunction(BaseConversationRepository.get_context)
    assert inspect.iscoroutinefunction(BaseConversationRepository.append_messages)


def test_get_context_contract_has_only_conversation_identifier() -> None:
    assert tuple(
        inspect.signature(BaseConversationRepository.get_context).parameters
    ) == ("self", "conversation_id")


def test_append_messages_contract_has_only_identifier_and_messages() -> None:
    assert tuple(
        inspect.signature(BaseConversationRepository.append_messages).parameters
    ) == ("self", "conversation_id", "messages")


def test_base_repository_has_no_instance_dictionary_contract() -> None:
    assert BaseConversationRepository.__slots__ == ()


def test_public_conversations_package_exports_repository_contracts() -> None:
    import tuesday.conversations as conversations

    assert conversations.BaseConversationRepository is BaseConversationRepository
    assert conversations.ConversationRepositoryError is ConversationRepositoryError
