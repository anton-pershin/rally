import json
import asyncio
from typing import TypedDict, Optional
import requests

from rich.progress import Progress
import aiohttp


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

    r = requests.post(
        llm_server_url,
        headers=headers,
        data=json.dumps(data),
    )

    response_json = json.loads(r.text)
    assert len(response_json["choices"]) == 1, "Only single message in choices is supported"

    return response_json["choices"][0]["message"]


async def _single_request_based_on_message_history_via_aiohttp(
    llm_server_url: str,
    message_history: list[LlmMessage],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
) -> LlmMessage:
    timeout = aiohttp.ClientTimeout()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        headers = {}
        if authorization is not None:
            headers["Authorization"] = authorization

        data = {
            "messages": message_history,
        }
        if model is not None:
            data["model"] = model

        async with session.post(
            llm_server_url,
            json=data,
            headers=headers
        ) as response:
            response_json = await response.json()
            assert len(response_json["choices"]) == 1, "Only single message in choices is supported"

            return response_json["choices"][0]["message"]


async def _single_request(
    llm_server_url: str,
    system_prompt: str,
    user_prompt: str,
    authorization: Optional[str] = None,
    model: Optional[str] = None,
) -> LlmMessage:
    message_history = make_up_message_history(
        system_prompt=system_prompt,
        user_prompt=user_prompt,  
    )

    return await _single_request_based_on_message_history_via_aiohttp(
        llm_server_url,
        message_history,
        authorization,
        model,
    )


async def _request_based_on_prompts(
    llm_server_url: str,
    system_prompt: str,
    user_prompts: list[str],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    progress_title: Optional[str] = None,
) -> str:
    if progress_title is not None:
        with Progress() as progress:
            task = progress.add_task(f"[cyan]{progress_title}", total=len(user_prompts))
            
            async def _request_with_progress(user_prompt: str):
                response = await _single_request(
                    llm_server_url=llm_server_url,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    authorization=authorization,
                    model=model,
                )
                progress.update(task, advance=1)
                return response

            responses = await asyncio.gather(
                *[_request_with_progress(user_prompt) for user_prompt in user_prompts]
            )
    else:
        responses = await asyncio.gather(
            *[
                _single_request(
                    llm_server_url=llm_server_url,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    authorization=authorization,
                    model=model,
                )
                for user_prompt in user_prompts
            ]
        )
    return responses


def request_based_on_prompts(
    llm_server_url: str,
    system_prompt: str,
    user_prompts: list[str],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
    progress_title: Optional[str] = None,
) -> list[str]:
    responses = asyncio.run(
        _request_based_on_prompts(
            llm_server_url,
            system_prompt,
            user_prompts, 
            authorization,
            model,
            progress_title,
        )
    )
    return [r["content"] for r in responses]


def request_based_on_message_history(
    llm_server_url: str,
    message_history: list[LlmMessage],
    authorization: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    message = _single_request_based_on_message_history(
        llm_server_url=llm_server_url,
        message_history=message_history,
        authorization=authorization,
        model=model,
    )

    return message

