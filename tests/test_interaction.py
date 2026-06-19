import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rally.interaction import (
    LlmMessage, _single_request_based_on_message_history,
    _single_request_based_on_message_history_via_aiohttp,
    make_up_message_history, request_based_on_message_history,
    request_based_on_prompts)


class TestMakeUpMessageHistory:
    def test_returns_system_and_user_messages(self) -> None:
        result = make_up_message_history(
            system_prompt="You are helpful.",
            user_prompt="What is 2+2?",
        )

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "What is 2+2?"}


class TestSingleRequestBasedOnMessageHistory:
    def test_basic_request(
        self, sample_message_history: list[LlmMessage], mock_llm_response: MagicMock
    ) -> None:
        response_data = mock_llm_response("4")

        with patch("rally.interaction.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.text = json.dumps(response_data)
            mock_post.return_value = mock_response

            result = _single_request_based_on_message_history(
                llm_server_url="http://localhost:8000/v1/chat",
                message_history=sample_message_history,
            )

            assert result == {"role": "assistant", "content": "4"}
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[0][0] == "http://localhost:8000/v1/chat"
            assert "Content-Type" in call_kwargs[1]["headers"]

    def test_with_authorization_and_model(
        self, sample_message_history: list[LlmMessage], mock_llm_response: MagicMock
    ) -> None:
        response_data = mock_llm_response("Response")

        with patch("rally.interaction.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.text = json.dumps(response_data)
            mock_post.return_value = mock_response

            _single_request_based_on_message_history(
                llm_server_url="http://localhost:8000/v1/chat",
                message_history=sample_message_history,
                authorization="Bearer test-token",
                model="gpt-4",
                max_output_tokens=512,
            )

            call_kwargs = mock_post.call_args
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer test-token"

            sent_data = json.loads(call_kwargs[1]["data"])
            assert sent_data["model"] == "gpt-4"
            assert sent_data["max_completion_tokens"] == 512
            assert sent_data["max_tokens"] == 512


class TestSingleRequestViaAiohttp:
    @pytest.mark.asyncio
    async def test_basic_request(
        self, sample_message_history: list[LlmMessage], mock_llm_response: MagicMock
    ) -> None:
        response_data = mock_llm_response("Async response")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=response_data)
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        result = await _single_request_based_on_message_history_via_aiohttp(
            session=mock_session,
            llm_server_url="http://localhost:8000/v1/chat",
            message_history=sample_message_history,
        )

        assert result == {"role": "assistant", "content": "Async response"}

    @pytest.mark.asyncio
    async def test_with_optional_params(
        self, sample_message_history: list[LlmMessage], mock_llm_response: MagicMock
    ) -> None:
        response_data = mock_llm_response("Response")

        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=response_data)
        mock_session.post = MagicMock(
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        )

        await _single_request_based_on_message_history_via_aiohttp(
            session=mock_session,
            llm_server_url="http://localhost:8000/v1/chat",
            message_history=sample_message_history,
            authorization="Bearer token",
            model="claude-3",
            max_output_tokens=1024,
        )

        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer token"
        sent_data = call_kwargs[1]["json"]
        assert sent_data["model"] == "claude-3"
        assert sent_data["max_completion_tokens"] == 1024


class TestRequestBasedOnPrompts:
    def test_returns_list_of_content_strings(
        self, mock_llm_response: MagicMock
    ) -> None:
        responses = [
            mock_llm_response("Answer 1"),
            mock_llm_response("Answer 2"),
            mock_llm_response("Answer 3"),
        ]

        with patch("rally.interaction.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )

            mock_responses = []
            for resp in responses:
                mock_resp = AsyncMock()
                mock_resp.json = AsyncMock(return_value=resp)
                mock_responses.append(mock_resp)

            mock_session.post = MagicMock(
                side_effect=[
                    AsyncMock(__aenter__=AsyncMock(return_value=mr))
                    for mr in mock_responses
                ]
            )

            result = request_based_on_prompts(
                llm_server_url="http://localhost:8000/v1/chat",
                max_concurrent_requests=2,
                system_prompt="You are helpful.",
                user_prompts=["Q1", "Q2", "Q3"],
            )

            assert result == ["Answer 1", "Answer 2", "Answer 3"]

    def test_with_progress_title(self, mock_llm_response: MagicMock) -> None:
        response = mock_llm_response("Answer")

        with patch("rally.interaction.aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )

            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=response)
            mock_session.post = MagicMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp))
            )

            result = request_based_on_prompts(
                llm_server_url="http://localhost:8000/v1/chat",
                max_concurrent_requests=2,
                system_prompt="You are helpful.",
                user_prompts=["Q1"],
                progress_title="Processing",
            )

            assert result == ["Answer"]


class TestRequestBasedOnMessageHistory:
    def test_returns_message_dict(
        self, sample_message_history: list[LlmMessage], mock_llm_response: MagicMock
    ) -> None:
        response_data = mock_llm_response("Hello back!")

        with patch("rally.interaction.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.text = json.dumps(response_data)
            mock_post.return_value = mock_response

            result = request_based_on_message_history(
                llm_server_url="http://localhost:8000/v1/chat",
                message_history=sample_message_history,
            )

            assert result == {"role": "assistant", "content": "Hello back!"}
