import pytest

import rally.interaction as interaction
from rally.interaction import make_up_message_history

REMOVED_REQUEST_FUNCTIONS = (
    "request_based_on_message_history",
    "request_based_on_prompts",
    "_request_based_on_prompts",
    "_single_request",
    "_single_request_based_on_message_history",
    "_single_request_based_on_message_history_via_aiohttp",
)


class TestMakeUpMessageHistory:
    def test_returns_system_and_user_messages(self) -> None:
        result = make_up_message_history(
            system_prompt="You are helpful.",
            user_prompt="What is 2+2?",
        )

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "What is 2+2?"}


class TestRemovedFunctions:
    @pytest.mark.parametrize("name", REMOVED_REQUEST_FUNCTIONS)
    def test_legacy_request_functions_are_gone(self, name: str) -> None:
        assert not hasattr(interaction, name)
