from typing import Optional, TypedDict


class LlmMessage(TypedDict):
    role: str
    content: str


def make_up_message_history(
    system_prompt: Optional[str],
    user_prompt: str,
) -> list[LlmMessage]:
    """The messages for one system/user pair, without a system message when absent."""
    messages: list[LlmMessage] = []
    if system_prompt is not None:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return messages
