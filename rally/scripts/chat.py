from hydra.utils import instantiate
from omegaconf import DictConfig
from rich.markdown import Markdown

import hydra
from rally.interaction import LlmMessage
from rally.utils.common import get_config_path
from rally.utils.console import console, prompt_user

CONFIG_NAME = "config_chat"


def chat(cfg: DictConfig) -> None:
    llm = instantiate(cfg.llm)

    messages: list[LlmMessage] = [{"role": "system", "content": cfg.system_prompt}]

    while True:
        user_input = prompt_user().strip()

        if user_input.lower() in cfg.stop_words:
            break

        try:
            messages.append({"role": "user", "content": user_input})

            assistant_message = llm.request(messages)
            if assistant_message is None:
                raise ValueError("Invalid response from the LLM.")

            messages.append(assistant_message)
            markdown = Markdown(assistant_message["content"])
            console.print(markdown)
            console.print()

        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")


if __name__ == "__main__":
    hydra.main(
        config_path=str(get_config_path()),
        config_name=CONFIG_NAME,
        version_base="1.3",
    )(chat)()
