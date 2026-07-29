# Spec: Enable/Disable Thinking

## 1. Requirement Analysis

The user wants to control whether the LLM server (vLLM or compatible) uses thinking mode at the **request level** by including or omitting `chat_template_kwargs` in the HTTP request body.

### Requirements

1. **`Llm` config object**: Add an `enable_thinking: Optional[bool]` attribute to the `Llm` base class so it is available to all subclasses (`LocalLlm`, `OpenAiApiLlmWithAuthorization`). Default is `None`.

2. **Request payload construction**: When `enable_thinking is not None`, the request body must include:
   ```json
   {"chat_template_kwargs": {"enable_thinking": <value>}}
   ```
   When `enable_thinking is None`, the field `chat_template_kwargs` must be **absent** from the request body entirely (let the server use its own default).

3. **Threading through `interaction.py`**: The `enable_thinking` parameter must be threaded through all request functions:
   - `_single_request_based_on_message_history` (sync)
   - `_single_request_based_on_message_history_via_aiohttp` (async)
   - `_single_request` (async wrapper)
   - `_request_based_on_prompts` (async batch)
   - `request_based_on_prompts` (sync batch)
   - `request_based_on_message_history` (sync public)

4. **Chat script wiring**: `chat.py` must pass `llm.enable_thinking` to `request_based_on_message_history`.

5. **Hydra config**: Add `enable_thinking` field to the `local.yaml` LLM config (and optionally to other LLM configs). The field defaults to `null` (None) when not specified.

6. **Post-processing independence**: `thinking.py` remains unchanged and independent. No coordination between server-side enable/disable and post-processing.

### Expected Variants

| `enable_thinking` value | Request body includes `chat_template_kwargs`? | Server behavior |
|---|---|---|
| `None` (default) | No | Server decides (its own default) |
| `True` | Yes: `{"enable_thinking": true}` | Thinking enabled |
| `False` | Yes: `{"enable_thinking": false}` | Thinking disabled |

## 2. Tests

### 2.1 `tests/test_llm.py` — Llm config tests

- **`test_enable_thinking_default_none`**: Construct `Llm` without `enable_thinking` → assert `llm.enable_thinking is None`.
- **`test_enable_thinking_set_true`**: Construct `Llm(enable_thinking=True)` → assert `llm.enable_thinking is True`.
- **`test_enable_thinking_set_false`**: Construct `Llm(enable_thinking=False)` → assert `llm.enable_thinking is False`.
- **`test_local_llm_enable_thinking`**: Construct `LocalLlm(enable_thinking=False)` → assert it propagates to the base class.
- **`test_openai_api_llm_enable_thinking`**: Construct `OpenAiApiLlmWithAuthorization(enable_thinking=True)` → assert it propagates.

### 2.2 `tests/test_interaction.py` — Request payload tests

- **`test_sync_request_no_chat_template_kwargs_when_none`**: Call `_single_request_based_on_message_history` with `enable_thinking=None` (or omitted) → assert `"chat_template_kwargs"` is **not** in the sent JSON.
- **`test_sync_request_enable_thinking_true`**: Call with `enable_thinking=True` → assert sent JSON contains `"chat_template_kwargs": {"enable_thinking": True}`.
- **`test_sync_request_enable_thinking_false`**: Call with `enable_thinking=False` → assert sent JSON contains `"chat_template_kwargs": {"enable_thinking": False}`.
- **`test_async_request_no_chat_template_kwargs_when_none`**: Same as above but for `_single_request_based_on_message_history_via_aiohttp`.
- **`test_async_request_enable_thinking_true`**: Same as above but async.
- **`test_async_request_enable_thinking_false`**: Same as above but async.
- **`test_request_based_on_message_history_passes_enable_thinking`**: Verify the public sync wrapper threads `enable_thinking` through.
- **`test_request_based_on_prompts_passes_enable_thinking`**: Verify the batch sync wrapper threads `enable_thinking` through.

## 3. Implementation Plan

### 3.1 Solution Design

The change is straightforward parameter threading:

1. Add `enable_thinking: Optional[bool] = None` to `Llm.__init__` and store it as an attribute.
2. Pass it through `LocalLlm.__init__` and `OpenAiApiLlmWithAuthorization.__init__` via `super().__init__()`.
3. In both sync and async request builders in `interaction.py`, add the `enable_thinking` parameter and conditionally include `chat_template_kwargs` in the request data dict.
4. Thread the parameter through all intermediate wrapper functions.
5. In `chat.py`, pass `llm.enable_thinking` when calling `request_based_on_message_history`.
6. Add `enable_thinking:` (defaulting to null) to `config/llm/local.yaml`.

### 3.2 Todo List

1. [ ] Write the tests (test_llm.py and test_interaction.py additions)
2. [ ] Run all tests and ensure the new tests fail
3. [ ] Add `enable_thinking` to `Llm.__init__` in `rally/llm.py`
4. [ ] Thread `enable_thinking` through `LocalLlm.__init__` and `OpenAiApiLlmWithAuthorization.__init__`
5. [ ] Add `enable_thinking` parameter and `chat_template_kwargs` logic to `_single_request_based_on_message_history` (sync)
6. [ ] Add `enable_thinking` parameter and `chat_template_kwargs` logic to `_single_request_based_on_message_history_via_aiohttp` (async)
7. [ ] Thread `enable_thinking` through `_single_request`, `_request_based_on_prompts`, `request_based_on_prompts`, `request_based_on_message_history`
8. [ ] Wire `llm.enable_thinking` in `chat.py`
9. [ ] Add `enable_thinking` field to `config/llm/local.yaml`
10. [ ] Run all tests and ensure they pass
11. [ ] Run linters (black, isort, pylint, mypy)

### 3.3 Modification Summary

| File | Action |
|------|--------|
| `rally/llm.py` | Modified: add `enable_thinking: Optional[bool]` to `Llm`, thread through `LocalLlm` and `OpenAiApiLlmWithAuthorization` |
| `rally/interaction.py` | Modified: add `enable_thinking` param to all 6 request functions; add `chat_template_kwargs` to request data when `enable_thinking is not None` |
| `rally/scripts/chat.py` | Modified: pass `llm.enable_thinking` to `request_based_on_message_history` |
| `config/llm/local.yaml` | Modified: add `enable_thinking:` field (null default) |
| `tests/test_llm.py` | Modified: add 5 tests for `enable_thinking` attribute on all Llm classes |
| `tests/test_interaction.py` | Modified: add 8 tests verifying `chat_template_kwargs` presence/absence in sync and async request payloads |
