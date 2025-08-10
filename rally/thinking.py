def remove_thinking_qwen2_5(response: str) -> str:
    return response


def remove_thinking_qwen3(response: str) -> str:
    # find the last occurence of \n and take all the symbols
    # located after this symbol
    # TODO: this must be wrong
    return response.split("\n")[-1]


def remove_thinking_qwq(response: str) -> str:
    return response.split("</think>")[-1].strip()


THINKING_REMOVERS = {
    "qwen2.5": remove_thinking_qwen2_5,
    "qwen3": remove_thinking_qwen3,
    "qwq": remove_thinking_qwq,
}

