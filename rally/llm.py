import asyncio
import json
import logging
from typing import Any, Optional

import aiohttp
import requests

from rally.interaction import LlmMessage, make_up_message_history


class Llm:
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        model_family: Optional[str] = None,
        authorization: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        self.url: str = url
        self.max_concurrent_requests: int = max_concurrent_requests
        self.model_family: Optional[str] = model_family
        self.authorization: Optional[str] = authorization
        self.model: Optional[str] = model
        self.max_output_tokens: Optional[int] = max_output_tokens
        self.enable_thinking: Optional[bool] = enable_thinking

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
    def __init__(
        self,
        url: str,
        max_concurrent_requests: int,
        model_family: str,
        max_output_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        super().__init__(
            url=url,
            max_concurrent_requests=max_concurrent_requests,
            model_family=model_family,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
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
    ) -> None:
        super().__init__(
            url=url,
            max_concurrent_requests=max_concurrent_requests,
            authorization=f"Bearer {api_key}",
            model=model,
            max_output_tokens=max_output_tokens,
            enable_thinking=enable_thinking,
        )
