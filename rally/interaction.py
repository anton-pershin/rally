import json
import asyncio
from typing import TypedDict, Optional
import requests

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
    async with aiohttp.ClientSession() as session:
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
) -> str:
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
) -> list[str]:
    responses = asyncio.run(
        _request_based_on_prompts(
            llm_server_url,
            system_prompt,
            user_prompts, 
            authorization, 
            model,
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

