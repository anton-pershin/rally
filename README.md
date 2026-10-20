# rally

Rally is a Python library for interaction with and performance tests of local and external LLMs.

## Features

- Send synchronous, asynchronous, and batch requests.
- Stream LLM responses as typed events.
- Configure LLM endpoints and request settings with Hydra.
- Build message histories from system and user prompts.

## User guide

### Use Rally as a library

Create an LLM client, build a message history, and send a request:

```python
from rally.interaction import make_up_message_history
from rally.llm import LocalLlm

llm = LocalLlm(
    url="http://localhost:9191/v1/chat/completions",
    max_concurrent_requests=16,
    model_family="qwen2.5",
)
messages = make_up_message_history(
    system_prompt="Answer clearly and concisely.",
    user_prompt="What is a language model?",
)

response = llm.request(messages)
if response is None:
    raise RuntimeError("The LLM returned an invalid response.")
print(response["content"])
```

The endpoint and model family must match your LLM server and model. `Llm` also provides `arequest()` for async requests, `request_batch()` and `arequest_batch()` for batches, and `stream()` for incremental events. See `rally/llm.py` for their signatures and return types.

### Use an interactive chat script

The chat script is a simple interactive client.

Start the chat script:

```sh
python rally/scripts/chat.py
```

The script uses `config/config_chat.yaml`. It loads the selected LLM configuration from `config/llm/` and user-specific settings from `config/user_settings/user_settings.yaml`.
Set `CAILA_API_KEY` to your Caila API key when using a Caila configuration. Set `RALLY_HYDRA_PATH` to the base directory for Hydra run outputs and logs; it defaults to `/tmp/rally_hydra`.


## Developer guide

### Run tests

```sh
pytest
```

### Development checks

Use type hints in Python code. Before completing code changes, run:

```sh
black rally/
isort rally/
pylint rally/
mypy rally/
```

### Project structure

- `rally/llm.py` — LLM settings and request operations.
- `rally/interaction.py` — message types and message-history construction.
- `rally/thinking.py` — model-family-specific reasoning-removal functions.
- `rally/scripts/` — executable scripts.
- `config/` — Hydra configuration.
- `.internal/` — SDD constitution, specifications, and terminology.

Each executable script has a Hydra configuration file named `config/config_<script>.yaml`. Keep configurable runtime settings in Hydra configuration rather than hard-coding them. Store user-specific values, such as API tokens and local paths, in `config/user_settings/user_settings.yaml`.
