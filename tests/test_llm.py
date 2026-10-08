import json
import logging
import socket
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from rally.llm import (
    Llm,
    LlmAuthorizationError,
    LlmError,
    LlmStreamRejectedError,
    LlmTimeoutError,
    LlmTransportError,
    LlmUsage,
    LocalLlm,
    OpenAiApiLlmWithAuthorization,
)
from tests.streaming_stub import (
    StreamingStub,
    broken_line,
    content_line,
    delta_line,
    done_line,
    finish_line,
    keepalive_line,
    reasoning_line,
    role_line,
    usage_line,
)


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


def free_port() -> int:
    """A port nothing is listening on, for the connection-refused case."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class TestStream:
    """The streamed request: FR1, FR8 (variants rows 14, 15, 16, 19)."""

    def test_stream_posts_built_headers_and_body_with_stream_keys(
        self, sample_message_history: list[Any]
    ) -> None:
        with StreamingStub(lines=[done_line()]) as stub:
            llm = make_llm(
                url=stub.url,
                model="qwen3-8b",
                max_output_tokens=100,
                enable_thinking=False,
            )
            list(llm.stream(sample_message_history))

        assert len(stub.requests) == 1
        received = stub.requests[0]
        assert received["headers"]["Content-Type"] == "application/json"
        assert "Authorization" not in received["headers"]
        assert received["body"] == {
            "messages": sample_message_history,
            "model": "qwen3-8b",
            "max_completion_tokens": 100,
            "max_tokens": 100,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def test_stream_body_omits_unconfigured_parameters(
        self, sample_message_history: list[Any]
    ) -> None:
        with StreamingStub(lines=[done_line()]) as stub:
            list(make_llm(url=stub.url).stream(sample_message_history))

        assert stub.requests[0]["body"] == {
            "messages": sample_message_history,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def test_stream_without_authorization_sends_no_authorization_header(
        self, sample_message_history: list[Any]
    ) -> None:
        with StreamingStub(lines=[done_line()]) as unauthorised:
            list(make_llm(url=unauthorised.url).stream(sample_message_history))
        with StreamingStub(lines=[done_line()]) as authorised:
            llm = make_llm(url=authorised.url, authorization="Bearer secret")
            list(llm.stream(sample_message_history))

        assert "Authorization" not in unauthorised.requests[0]["headers"]
        assert authorised.requests[0]["headers"]["Authorization"] == "Bearer secret"


class TestStreamEvents:
    """Event delivery: FR2-FR6 (variants rows 1-7, 20)."""

    def test_content_chunks_yield_one_event_each_in_order(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("Al"), content_line("ice"), done_line()]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        contents = [event.content for event in events if event.content]
        assert contents == ["Al", "ice"]
        assert "".join(contents) == "Alice"

    def test_usage_is_visible_with_both_token_counts(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("hi"), usage_line(12, 5), done_line()]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        usages = [event.usage for event in events if event.usage is not None]
        assert usages == [LlmUsage(prompt_tokens=12, completion_tokens=5)]

    def test_absent_usage_leaves_usage_unset_and_content_events_countable(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("a"), content_line("b"), done_line()]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        assert all(event.usage is None for event in events)
        assert len([event for event in events if event.content]) == 2

    def test_reasoning_is_separate_from_content(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [reasoning_line("let me think"), content_line("ok"), done_line()]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        reasoning_events = [event for event in events if event.reasoning]
        assert [event.reasoning for event in reasoning_events] == ["let me think"]
        assert reasoning_events[0].content is None
        assert "".join(event.content or "" for event in events) == "ok"

    def test_reasoning_reads_the_key_the_provider_sent(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [
            delta_line({"reasoning_content": "", "reasoning": "other text"}),
            delta_line({"reasoning": "alternate spelling"}),
            done_line(),
        ]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        # The key the provider sent decides, even when it is empty: an empty
        # delta must not borrow the text of the other spelling.
        assert events[0].reasoning == ""
        assert events[1].reasoning == "alternate spelling"

    def test_contentless_chunk_yields_event_without_content(
        self, sample_message_history: list[Any]
    ) -> None:
        with StreamingStub(lines=[role_line(), done_line()]) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        assert len(events) == 1
        assert events[0].content is None

    def test_end_marker_yields_final_event_and_closes_iteration(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [
            content_line("x"),
            finish_line("stop"),
            done_line(),
            content_line("never"),
        ]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        assert [event.content for event in events if event.content] == ["x"]
        assert [event.finish_reason for event in events if event.finish_reason] == [
            "stop"
        ]

    def test_stream_without_end_marker_ends_at_response_close(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("a"), content_line("b")]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        assert [event.content for event in events if event.content] == ["a", "b"]
        assert all(event.finish_reason is None for event in events)

    def test_events_arrive_incrementally_not_buffered(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("a"), content_line("b"), content_line("c")]
        with StreamingStub(lines=lines, delay=0.15) as stub:
            arrivals = []
            for event in make_llm(url=stub.url).stream(sample_message_history):
                if event.content:
                    arrivals.append(time.perf_counter())

        assert len(arrivals) == 3
        # A buffering implementation delivers every event together, so the gap
        # between the first and the last content event collapses to ~0.
        assert arrivals[-1] - arrivals[0] > 0.1

    def test_non_chunk_lines_are_skipped(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [
            keepalive_line(),
            content_line("a"),
            broken_line(),
            content_line("b"),
            done_line(),
        ]
        with StreamingStub(lines=lines) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        assert [event.content for event in events if event.content] == ["a", "b"]


class TestStreamFailures:
    """Failures and the timeout: FR7, FR9 (variants rows 8-13)."""

    @pytest.mark.parametrize("status", [401, 403])
    def test_authorization_failure_is_its_own_error(
        self, sample_message_history: list[Any], status: int
    ) -> None:
        with StreamingStub(status=status) as stub:
            with pytest.raises(LlmAuthorizationError):
                list(make_llm(url=stub.url).stream(sample_message_history))

    def test_rejected_stream_raises_with_status(
        self, sample_message_history: list[Any]
    ) -> None:
        with StreamingStub(status=500, body=b'{"error": "boom"}') as stub:
            with pytest.raises(LlmStreamRejectedError) as failure:
                list(make_llm(url=stub.url).stream(sample_message_history))

        assert failure.value.status == 500

    def test_transport_failure_before_content(
        self, sample_message_history: list[Any]
    ) -> None:
        url = f"http://127.0.0.1:{free_port()}/v1/chat/completions"
        with pytest.raises(LlmTransportError):
            list(make_llm(url=url).stream(sample_message_history))

    def test_transport_failure_mid_stream_after_delivered_events(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("a"), content_line("b")]
        with StreamingStub(lines=lines, mode="abrupt", abrupt_after=1) as stub:
            delivered = []
            with pytest.raises(LlmTransportError):
                for event in make_llm(url=stub.url).stream(sample_message_history):
                    if event.content:
                        delivered.append(event.content)

        assert delivered == ["a"]

    def test_stream_with_no_content_is_not_an_error(
        self, sample_message_history: list[Any]
    ) -> None:
        with StreamingStub(lines=[done_line()]) as stub:
            events = list(make_llm(url=stub.url).stream(sample_message_history))

        assert events == []

    @pytest.mark.parametrize("mode", ["silent", "silent_after_headers"])
    def test_timeout_is_its_own_error_and_reaches_every_constructor(
        self, sample_message_history: list[Any], mode: str
    ) -> None:
        with StreamingStub(mode=mode) as stub:
            llm = make_llm(url=stub.url, timeout=0.3)
            with pytest.raises(LlmTimeoutError):
                list(llm.stream(sample_message_history))

        assert make_llm().timeout is None
        assert (
            LocalLlm(
                url="http://localhost:9191/v1/chat/completions",
                max_concurrent_requests=1,
                model_family="qwen3",
                timeout=5.0,
            ).timeout
            == 5.0
        )
        assert (
            OpenAiApiLlmWithAuthorization(
                url="https://example.invalid/v1/chat/completions",
                max_concurrent_requests=1,
                api_key="key",
                model="m",
                timeout=5.0,
            ).timeout
            == 5.0
        )

    def test_failure_types_are_distinguishable(self) -> None:
        assert issubclass(LlmAuthorizationError, LlmError)
        assert issubclass(LlmStreamRejectedError, LlmError)
        assert issubclass(LlmTransportError, LlmError)
        assert issubclass(LlmTimeoutError, LlmTransportError)
        assert not issubclass(LlmAuthorizationError, LlmStreamRejectedError)
        assert not isinstance(LlmTransportError("gone"), LlmTimeoutError)
        assert LlmStreamRejectedError(status=418, message="teapot").status == 418


class TestStreamLifecycle:
    """Early stop releases the response: FR11 (variant row 17)."""

    def test_early_stop_releases_the_response(
        self, sample_message_history: list[Any]
    ) -> None:
        lines = [content_line("a"), content_line("b"), content_line("c")]
        released: list[str] = []
        real_post = requests.post

        def _recording_post(*args: Any, **kwargs: Any) -> Any:
            response = real_post(*args, **kwargs)
            original_close = response.close

            def _close() -> None:
                released.append("closed")
                original_close()

            response.close = _close
            return response

        with StreamingStub(lines=lines, delay=0.05) as stub:
            with patch("rally.llm.requests.post", side_effect=_recording_post):
                events = make_llm(url=stub.url).stream(sample_message_history)
                first = [next(events), next(events)]
                events.close()

        assert [event.content for event in first] == ["a", "b"]
        assert released == ["closed"]


class TestNonStreamingUnchanged:
    """The non-streaming surface is untouched: FR10 (variant row 18)."""

    def test_request_returns_none_instead_of_raising(
        self, sample_message_history: list[Any]
    ) -> None:
        body = json.dumps({"choices": []}).encode("utf-8")
        with StreamingStub(mode="json", body=body) as stub:
            llm = make_llm(url=stub.url, model="m", authorization="Bearer secret")
            assert llm.request(sample_message_history) is None
            assert stub.requests[0]["body"] == {
                "messages": sample_message_history,
                "model": "m",
            }
