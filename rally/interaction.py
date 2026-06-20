import asyncio
import json
from typing import Optional, TypedDict

import aiohttp
import requests
from rich.progress import Progress


class LlmMessage(TypedDict):
    role: str
    content: str


def make_up_message_history(
    system_prompt: str,
    user_prompt: str,
) -> list[LlmMessage]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def _single_request_based_on_message_history(
    llm_server_url: str,
    message_history: list[LlmMessage],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> LlmMessage:
    headers = {
        "Content-Type": "application/json",
    }
    if authorization is not None:
        headers["Authorization"] = authorization

    data = {
        "messages": message_history,
    }
    if model is not None:
        data["model"] = model
    if max_output_tokens is not None:
        data["max_completion_tokens"] = max_output_tokens
        data["max_tokens"] = max_output_tokens
    if enable_thinking is not None:
        data["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    r = requests.post(
        llm_server_url,
        headers=headers,
        data=json.dumps(data),
    )

    response_json = json.loads(r.text)
    assert (
        len(response_json["choices"]) == 1
    ), "Only single message in choices is supported"

    return response_json["choices"][0]["message"]


async def _single_request_based_on_message_history_via_aiohttp(
    session: aiohttp.ClientSession,
    llm_server_url: str,
    message_history: list[LlmMessage],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> LlmMessage:
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization

    data = {
        "messages": message_history,
    }
    if model is not None:
        data["model"] = model
    if max_output_tokens is not None:
        data["max_completion_tokens"] = max_output_tokens
    if enable_thinking is not None:
        data["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    async with session.post(llm_server_url, json=data, headers=headers) as response:
        response_json = await response.json()
        assert (
            len(response_json["choices"]) == 1
        ), "Only single message in choices is supported"

        return response_json["choices"][0]["message"]


async def _single_request(
    session: aiohttp.ClientSession,
    llm_server_url: str,
    system_prompt: str,
    user_prompt: str,
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> LlmMessage:
    message_history = make_up_message_history(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return await _single_request_based_on_message_history_via_aiohttp(
        session,
        llm_server_url,
        message_history,
        authorization,
        model,
        max_output_tokens,
        enable_thinking,
    )


async def _request_based_on_prompts(
    llm_server_url: str,
    max_concurrent_requests: int,
    system_prompt: str,
    user_prompts: list[str],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    progress_title: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    timeout = aiohttp.ClientTimeout()
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        if progress_title is not None:
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]{progress_title}", total=len(user_prompts)
                )

                async def _request_with_progress(user_prompt: str):
                    response = await _single_request(
                        session=session,
                        llm_server_url=llm_server_url,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        authorization=authorization,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        enable_thinking=enable_thinking,
                    )
                    progress.update(task, advance=1)
                    return response

                responses = await asyncio.gather(
                    *[
                        _request_with_progress(user_prompt)
                        for user_prompt in user_prompts
                    ]
                )
        else:
            responses = await asyncio.gather(
                *[
                    _single_request(
                        session=session,
                        llm_server_url=llm_server_url,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        authorization=authorization,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        enable_thinking=enable_thinking,
                    )
                    for user_prompt in user_prompts
                ]
            )
    return responses


def request_based_on_prompts(
    llm_server_url: str,
    max_concurrent_requests: int,
    system_prompt: str,
    user_prompts: list[str],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    progress_title: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> list[str]:
    responses = asyncio.run(
        _request_based_on_prompts(
            llm_server_url,
            max_concurrent_requests,
            system_prompt,
            user_prompts,
            authorization,
            model,
            progress_title,
            max_output_tokens,
            enable_thinking,
        )
    )
    return [r["content"] for r in responses]


def request_based_on_message_history(
    llm_server_url: str,
    message_history: list[LlmMessage],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> str:
    message = _single_request_based_on_message_history(
        llm_server_url=llm_server_url,
        message_history=message_history,
        authorization=authorization,
        model=model,
        max_output_tokens=max_output_tokens,
        enable_thinking=enable_thinking,
    )

    return message
