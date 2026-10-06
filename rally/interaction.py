from typing import TypedDict


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
