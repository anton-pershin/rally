import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import aiohttp
import requests

from rally.interaction import LlmMessage, make_up_message_history


@dataclass(frozen=True)
class LlmUsage:
    """The token counts a server reports for one completion."""

    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


@dataclass(frozen=True)
class LlmStreamEvent:
    """One typed event of a streamed completion.

    ``content`` and ``reasoning`` carry the deltas of the chunk that produced
    the event, ``usage`` the report when the server sends one, and
    ``finish_reason`` the server's declaration of why the completion finished.
    There is deliberately no "finished" flag: the iteration ending already says
    the stream is over, and only the finish reason answers whether the server
    declared the completion complete. No provider field name is exposed here on
    purpose: translating the provider's dialect is this module's job, not the
    caller's.
    """

    content: Optional[str] = None
    reasoning: Optional[str] = None
    usage: Optional[LlmUsage] = None
    finish_reason: Optional[str] = None


class LlmError(Exception):
    """Base class for the failures the streaming operation raises."""


class LlmAuthorizationError(LlmError):
    """The server rejected the credentials (HTTP 401 or 403)."""


class LlmStreamRejectedError(LlmError):
    """The server refused the streaming request with another HTTP status."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"Streaming request rejected with HTTP {status}")
        self.status = status


class LlmTransportError(LlmError):
    """The connection failed, or the stream was dropped before it finished."""


class LlmTimeoutError(LlmTransportError):
    """No data arrived within the configured timeout."""


_TIMEOUT_NAMES = {"ReadTimeoutError", "ConnectTimeoutError", "TimeoutError"}


def _raised_by_timeout(error: BaseException) -> bool:
    """True when the failure chain holds a timeout, however it is wrapped.

    ``requests`` reports a read timeout that happens while a streamed body is
    being read as a ``ConnectionError``, so the cause chain has to be walked to
    tell a stalled server from a connection that was dropped.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (requests.exceptions.Timeout, TimeoutError)):
            return True
        if type(current).__name__ in _TIMEOUT_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def _stream_failure(error: BaseException) -> LlmError:
    """The typed failure for a transport-level error."""
    if _raised_by_timeout(error):
        return LlmTimeoutError(f"No data arrived within the timeout: {error}")
    return LlmTransportError(f"The streaming connection failed: {error}")


class Llm:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        model_family: Optional[str] = None,
        authorization: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.url: str = url
        self.max_concurrent_requests: int = max_concurrent_requests
        self.model_family: Optional[str] = model_family
        self.authorization: Optional[str] = authorization
        self.model: Optional[str] = model
        self.max_output_tokens: Optional[int] = max_output_tokens
        self.enable_thinking: Optional[bool] = enable_thinking
        self.timeout: Optional[float] = timeout

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.authorization is not None:
            headers["Authorization"] = self.authorization

        return headers

    def build_payload(self, message_history: list[LlmMessage]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": message_history,
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.max_output_tokens is not None:
            payload["max_completion_tokens"] = self.max_output_tokens
            payload["max_tokens"] = self.max_output_tokens
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        return payload

    def request(self, message_history: list[LlmMessage]) -> Optional[LlmMessage]:
        response = requests.post(
            self.url,
            headers=self.build_headers(),
            data=json.dumps(self.build_payload(message_history)),
        )

        response_json = json.loads(response.text)
        if ("choices" not in response_json) or (len(response_json["choices"]) != 1):
            logging.error("Invalid response %s", str(response_json))
            return None

        return response_json["choices"][0]["message"]

    def stream(self, message_history: list[LlmMessage]) -> Iterator[LlmStreamEvent]:
        """Send a streaming request and yield its events as they arrive.

        The request is the same request description as ``request`` sends, plus
        the two keys that ask for a stream and for the usage report. Events are
        yielded as their chunks arrive, so a caller can time them; the response
        is released when the iteration ends or the caller stops it early.
        """
        payload = self.build_payload(message_history)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        try:
            response = requests.post(
                self.url,
                headers=self.build_headers(),
                data=json.dumps(payload),
                stream=True,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as err:
            raise _stream_failure(err) from err

        with response:
            if response.status_code in (401, 403):
                raise LlmAuthorizationError(
                    f"Authorization failed (HTTP {response.status_code}) "
                    f"for {self.url}. Check the API key / credentials configuration."
                )
            if not 200 <= response.status_code < 300:
                raise LlmStreamRejectedError(
                    status=response.status_code,
                    message=(
                        f"Streaming request rejected (HTTP {response.status_code})"
                        f" for {self.url}."
                    ),
                )
            yield from self._iter_events(response)

    def _iter_events(self, response: "requests.Response") -> Iterator[LlmStreamEvent]:
        """Translate the server's chunks into typed events, in arrival order."""
        try:
            for raw_line in response.iter_lines():
                line = (raw_line or b"").decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue  # blank keep-alive and comment lines carry nothing
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # a truncated or non-JSON line is not a chunk
                yield self._event_from_chunk(chunk)
        except requests.exceptions.RequestException as err:
            raise _stream_failure(err) from err

    @staticmethod
    def _event_from_chunk(chunk: Any) -> LlmStreamEvent:
        """One typed event for one provider chunk, whose fields may be absent."""
        usage = chunk.get("usage")
        choices = chunk.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}

        return LlmStreamEvent(
            content=delta.get("content"),
            reasoning=delta.get("reasoning_content") or delta.get("reasoning"),
            finish_reason=choice.get("finish_reason"),
            usage=(
                LlmUsage(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
                if usage
                else None
            ),
        )

    async def arequest(self, message_history: list[LlmMessage]) -> Optional[LlmMessage]:
        async with aiohttp.ClientSession() as session:
            return await self._arequest_with_session(
                session=session,
                message_history=message_history,
            )

    async def arequest_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        progress_title: Optional[str] = None,
    ) -> list[Optional[str]]:
        timeout = aiohttp.ClientTimeout()
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests)

        if progress_title is not None:
            logging.info("Starting %s (%d requests)", progress_title, len(user_prompts))

        completed = 0

        async def _request(user_prompt: str) -> Optional[LlmMessage]:
            nonlocal completed
            response = await self._arequest_with_session(
                session=session,
                message_history=make_up_message_history(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ),
            )
            if progress_title is not None:
                completed += 1
                logging.debug(
                    "%s: %d/%d done", progress_title, completed, len(user_prompts)
                )
            return response

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            responses = await asyncio.gather(
                *[_request(user_prompt) for user_prompt in user_prompts]
            )

        if progress_title is not None:
            logging.info("Completed %s", progress_title)

        return [
            response["content"] if response is not None else None
            for response in responses
        ]

    def request_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        progress_title: Optional[str] = None,
    ) -> list[Optional[str]]:
        return asyncio.run(
            self.arequest_batch(
                system_prompt=system_prompt,
                user_prompts=user_prompts,
                progress_title=progress_title,
            )
        )

    async def _arequest_with_session(
        self,
        session: aiohttp.ClientSession,
        message_history: list[LlmMessage],
    ) -> Optional[LlmMessage]:
        async with session.post(
            self.url,
            json=self.build_payload(message_history),
            headers=self.build_headers(),
        ) as response:
            response_json = await response.json()
            if ("choices" not in response_json) or (len(response_json["choices"]) != 1):
                logging.error("Invalid response %s", str(response_json))
                return None

            message = response_json["choices"][0]["message"]
            if "reasoning_content" in message:
                logging.warning("Reasoning content found in response")

            return message


class LocalLlm(Llm):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        model_family: str,
        max_output_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            url=url,
            max_concurrent_requests=max_concurrent_requests,
            model_family=model_family,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )


class OpenAiApiLlmWithAuthorization(Llm):
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        api_key: str,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(
            url=url,
            max_concurrent_requests=max_concurrent_requests,
            authorization=f"Bearer {api_key}",
            model=model,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
