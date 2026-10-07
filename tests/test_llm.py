import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rally.llm import Llm, LocalLlm, OpenAiApiLlmWithAuthorization


class TestLlm:
    def test_stores_all_attributes(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
            model_family="qwen",
            authorization="Bearer token123",
            model="qwen2.5-7b",
            max_output_tokens=512,
        )

        assert llm.url == "http://localhost:8000"
        assert llm.max_concurrent_requests == 10
        assert llm.model_family == "qwen"
        assert llm.authorization == "Bearer token123"
        assert llm.model == "qwen2.5-7b"
        assert llm.max_output_tokens == 512

    def test_optional_defaults_to_none(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=5,
        )

        assert llm.model_family is None
        assert llm.authorization is None
        assert llm.model is None
        assert llm.max_output_tokens is None


class TestLocalLlm:
    def test_passes_model_family_as_required(self) -> None:
        llm = LocalLlm(
            url="http://localhost:8080",
            max_concurrent_requests=4,
            model_family="llama",
        )

        assert llm.url == "http://localhost:8080"
        assert llm.max_concurrent_requests == 4
        assert llm.model_family == "llama"
        assert llm.max_output_tokens is None

    def test_with_max_output_tokens(self) -> None:
        llm = LocalLlm(
            url="http://localhost:8080",
            max_concurrent_requests=4,
            model_family="llama",
            max_output_tokens=1024,
        )

        assert llm.max_output_tokens == 1024


class TestOpenAiApiLlmWithAuthorization:
    def test_formats_bearer_token(self) -> None:
        llm = OpenAiApiLlmWithAuthorization(
            url="https://api.openai.com/v1/chat/completions",
            max_concurrent_requests=8,
            api_key="sk-test-key",
            model="gpt-4",
        )

        assert llm.authorization == "Bearer sk-test-key"
        assert llm.model == "gpt-4"
        assert llm.max_output_tokens is None

    def test_with_max_output_tokens(self) -> None:
        llm = OpenAiApiLlmWithAuthorization(
            url="https://api.openai.com/v1/chat/completions",
            max_concurrent_requests=8,
            api_key="sk-test-key",
            model="gpt-4",
            max_output_tokens=2048,
        )

        assert llm.max_output_tokens == 2048


class TestEnableThinking:
    def test_enable_thinking_default_none(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
        )

        assert llm.enable_thinking is None

    def test_enable_thinking_set_true(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
            enable_thinking=True,
        )

        assert llm.enable_thinking is True

    def test_enable_thinking_set_false(self) -> None:
        llm = Llm(
            url="http://localhost:8000",
            max_concurrent_requests=10,
            enable_thinking=False,
        )

        assert llm.enable_thinking is False

    def test_local_llm_enable_thinking(self) -> None:
        llm = LocalLlm(
            url="http://localhost:8080",
            max_concurrent_requests=4,
            model_family="qwen3",
            enable_thinking=False,
        )

        assert llm.enable_thinking is False

    def test_openai_api_llm_enable_thinking(self) -> None:
        llm = OpenAiApiLlmWithAuthorization(
            url="https://api.openai.com/v1/chat/completions",
            max_concurrent_requests=8,
            api_key="sk-test-key",
            model="gpt-4",
            enable_thinking=True,
        )

        assert llm.enable_thinking is True


MESSAGES = [{"role": "user", "content": "Hello"}]


def make_llm(**overrides: Any) -> Llm:
    params: dict[str, Any] = {
        "url": "http://localhost:8000/v1/chat/completions",
        "max_concurrent_requests": 16,
    }
    params.update(overrides)
    return Llm(**params)


def make_response(content: str = "Test response") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakePostContext:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def __aenter__(self) -> "FakeAsyncResponse":
        return FakeAsyncResponse(self._payload)

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class FakeAsyncResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


class FakeAiohttpSession:
    """Stand-in for aiohttp.ClientSession recording the posted requests."""

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = list(payloads)
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> "FakeAiohttpSession":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def post(self, url: str, json: dict, headers: dict) -> FakePostContext:
        self.requests.append({"url": url, "json": json, "headers": headers})
        payload = self._payloads.pop(0) if self._payloads else {"choices": []}
        return FakePostContext(payload)


class TestBuildHeaders:
    def test_content_type_only_without_authorization(self) -> None:
        llm = make_llm()

        assert llm.build_headers() == {"Content-Type": "application/json"}

    def test_authorization_header_added(self) -> None:
        llm = make_llm(authorization="Bearer token123")

        assert llm.build_headers() == {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }


class TestBuildPayload:
    def test_model_key_present_and_absent(self) -> None:
        assert make_llm().build_payload(MESSAGES) == {"messages": MESSAGES}
        assert make_llm(model="qwen3-8b").build_payload(MESSAGES) == {
            "messages": MESSAGES,
            "model": "qwen3-8b",
        }

    def test_max_output_tokens_none_sends_neither_key(self) -> None:
        body = make_llm().build_payload(MESSAGES)

        assert "max_completion_tokens" not in body
        assert "max_tokens" not in body

    def test_max_output_tokens_set_sends_both_keys(self) -> None:
        body = make_llm(max_output_tokens=512).build_payload(MESSAGES)

        assert body["max_completion_tokens"] == 512
        assert body["max_tokens"] == 512

    def test_enable_thinking_true_and_false(self) -> None:
        assert make_llm(enable_thinking=True).build_payload(MESSAGES)[
            "chat_template_kwargs"
        ] == {"enable_thinking": True}
        assert make_llm(enable_thinking=False).build_payload(MESSAGES)[
            "chat_template_kwargs"
        ] == {"enable_thinking": False}

    def test_enable_thinking_none_omits_key(self) -> None:
        assert "chat_template_kwargs" not in make_llm().build_payload(MESSAGES)

    def test_all_optional_fields_none_body_is_messages_only(self) -> None:
        assert make_llm().build_payload(MESSAGES) == {"messages": MESSAGES}


class TestRequest:
    def test_request_posts_built_headers_and_body(self) -> None:
        llm = make_llm(
            authorization="Bearer token123",
            model="qwen3-8b",
            max_output_tokens=128,
            enable_thinking=False,
        )
        expected_body = {
            "messages": MESSAGES,
            "model": "qwen3-8b",
            "max_completion_tokens": 128,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        expected_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }

        with patch("rally.llm.requests.post") as mock_post:
            mock_post.return_value = MagicMock(text=json.dumps(make_response("4")))

            result = llm.request(MESSAGES)

        assert result == {"role": "assistant", "content": "4"}
        call = mock_post.call_args
        assert call[0][0] == llm.url
        assert call[1]["headers"] == expected_headers
        assert json.loads(call[1]["data"]) == expected_body

    @pytest.mark.parametrize(
        "payload",
        [
            {"choices": []},
            {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]},
        ],
    )
    def test_request_returns_none_on_invalid_choices(
        self, payload: dict, caplog: pytest.LogCaptureFixture
    ) -> None:
        with patch("rally.llm.requests.post") as mock_post:
            mock_post.return_value = MagicMock(text=json.dumps(payload))

            assert make_llm().request(MESSAGES) is None

        assert any("Invalid response" in r.getMessage() for r in caplog.records)


class TestArequest:
    @pytest.mark.anyio
    async def test_arequest_matches_request_headers_and_body(self) -> None:
        llm = make_llm(
            authorization="Bearer token123",
            model="qwen3-8b",
            max_output_tokens=128,
            enable_thinking=False,
        )
        session = FakeAiohttpSession([make_response("4")])

        with patch("rally.llm.aiohttp.ClientSession", return_value=session):
            result = await llm.arequest(MESSAGES)

        with patch("rally.llm.requests.post") as mock_post:
            mock_post.return_value = MagicMock(text=json.dumps(make_response("4")))
            llm.request(MESSAGES)
        sync_call = mock_post.call_args[1]

        assert result == {"role": "assistant", "content": "4"}
        assert session.requests[0]["url"] == llm.url
        assert session.requests[0]["headers"] == sync_call["headers"]
        assert session.requests[0]["json"] == json.loads(sync_call["data"])

    @pytest.mark.anyio
    async def test_none_enable_thinking_omits_chat_template_kwargs(self) -> None:
        for value, expected in (
            (None, None),
            (True, {"enable_thinking": True}),
            (False, {"enable_thinking": False}),
        ):
            session = FakeAiohttpSession([make_response()])

            with patch("rally.llm.aiohttp.ClientSession", return_value=session):
                await make_llm(enable_thinking=value).arequest(MESSAGES)

            sent = session.requests[0]["json"]
            if expected is None:
                assert "chat_template_kwargs" not in sent
            else:
                assert sent["chat_template_kwargs"] == expected

    @pytest.mark.anyio
    async def test_max_output_tokens_sends_both_keys(self) -> None:
        session = FakeAiohttpSession([make_response()])

        with patch("rally.llm.aiohttp.ClientSession", return_value=session):
            await make_llm(max_output_tokens=256).arequest(MESSAGES)

        sent = session.requests[0]["json"]
        assert sent["max_completion_tokens"] == 256
        assert sent["max_tokens"] == 256


class TestRequestBatch:
    def test_contents_in_prompt_order_and_progress_logging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = FakeAiohttpSession(
            [make_response("A1"), make_response("A2"), make_response("A3")]
        )

        with caplog.at_level(logging.DEBUG, logger="root"):
            with patch("rally.llm.aiohttp.ClientSession", return_value=session):
                contents = make_llm().request_batch(
                    system_prompt="You are helpful.",
                    user_prompts=["Q1", "Q2", "Q3"],
                    progress_title="Processing",
                )

        assert contents == ["A1", "A2", "A3"]
        assert session.requests[0]["json"]["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Q1"},
        ]
        messages = [r.getMessage() for r in caplog.records]
        assert "Starting Processing (3 requests)" in messages
        assert "Completed Processing" in messages
        assert "Processing: 3/3 done" in messages

    def test_no_logging_without_progress_title(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = FakeAiohttpSession([make_response("A1")])

        with caplog.at_level(logging.DEBUG):
            with patch("rally.llm.aiohttp.ClientSession", return_value=session):
                contents = make_llm().request_batch(
                    system_prompt="You are helpful.",
                    user_prompts=["Q1"],
                )

        assert contents == ["A1"]
        assert [r for r in caplog.records if r.name.startswith("rally")] == []


class TestArequestBatch:
    @pytest.mark.anyio
    async def test_connector_limit_is_max_concurrent_requests(self) -> None:
        session = FakeAiohttpSession([make_response("A1"), make_response("A2")])

        with patch("rally.llm.aiohttp.ClientSession", return_value=session):
            with patch("rally.llm.aiohttp.TCPConnector") as mock_connector:
                contents = await make_llm(max_concurrent_requests=7).arequest_batch(
                    system_prompt="You are helpful.",
                    user_prompts=["Q1", "Q2"],
                )

        assert contents == ["A1", "A2"]
        assert mock_connector.call_args[1]["limit"] == 7

    @pytest.mark.anyio
    async def test_none_at_failed_prompt_position(self) -> None:
        session = FakeAiohttpSession(
            [make_response("A1"), {"choices": []}, make_response("A3")]
        )

        with patch("rally.llm.aiohttp.ClientSession", return_value=session):
            contents = await make_llm().arequest_batch(
                system_prompt="You are helpful.",
                user_prompts=["Q1", "Q2", "Q3"],
            )

        assert contents == ["A1", None, "A3"]
