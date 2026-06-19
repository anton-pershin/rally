from typing import Callable

import pytest

from rally.interaction import LlmMessage


@pytest.fixture
def mock_llm_response() -> Callable:
    def _factory(content: str = "Test response") -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ]
        }

    return _factory


@pytest.fixture
def sample_message_history() -> list[LlmMessage]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
