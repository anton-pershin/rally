# Spec 02 — Standard logging for the library

## 1. Requirement analysis

The `rally` package is primarily a library, but it currently mixes `print()`,
`console.print()` and `rich` output statements. Higher-level applications that
depend on it rely on standard Python `logging`. This spec converts the one piece
of library code that forces terminal UI on callers — the rich `Progress` bar in
`rally/interaction.py` — to standard `logging`. All other output statements are
intentionally kept as-is (see decisions below).

### Requirements

- **R1 — No rich terminal UI in library code.** `rally/interaction.py` must not
  import or use `rich.progress.Progress` (or any other `rich` rendering) for
  progress reporting.

- **R2 — Progress reporting uses `logging`.** Progress reporting in
  `_request_based_on_prompts` must go through the standard `logging` module
  using its module-level convenience functions (`logging.info()`,
  `logging.debug()`) — no module-level `getLogger(__name__)` instance is created.

- **R3 — `progress_title` remains the on/off toggle.** When `progress_title` is
  `None`, no progress log messages are emitted (preserving the current "silent"
  semantics). When `progress_title` is provided, progress is logged using it as a
  label.

- **R4 — Log granularity:**
  - **INFO** — a start message and a completion message:
    - Start: message rendered as `Starting <progress_title> (<N> requests)` where
      `<N>` is `len(user_prompts)`.
    - Completion: message rendered as `Completed <progress_title>`.
  - **DEBUG** — one message per completed request, rendered as
    `<progress_title>: <i>/<N> done` where `<i>` is the running count of
    completed requests (not the prompt index) and `<N>` is the total count.

- **R5 — Unchanged output statements.** The following are intentionally kept
  exactly as they are and must NOT be converted to logging:
  - `rally/utils/console.py` — the `console` singleton and `prompt_user()`
    (interactive stdin input + trailing newline). Input cannot be replaced by
    logging, and the singleton remains a shared UI utility for scripts.
  - `rally/scripts/chat.py` — `console.print(Markdown(...))` rendering of the
    assistant reply (primary user-facing output of the chat CLI) and
    `console.print(f"[bold red]Error:[/] {str(e)}")` for caught errors (kept as
    styled console output).

- **R6 — No breaking API changes.** The public signatures of
  `request_based_on_prompts` and `_request_based_on_prompts` (including the
  `progress_title` parameter) remain unchanged.

- **R7 — Behavior preserved.** `request_based_on_prompts` still returns the
  list of response `content` strings in the same order; the logging change must
  not alter request execution or return values.

## 2. Implementation plan

### 2.1. Solution design

Add a plain `import logging` at the top of `rally/interaction.py` (no
module-level `getLogger(__name__)` — the `logging` module's convenience
functions are used directly). Replace the
`if progress_title is not None: with Progress() ... else: ...` block in
`_request_based_on_prompts` with a single unified request path that
conditionally logs:

```python
async def _request_based_on_prompts(...) -> str:
    timeout = aiohttp.ClientTimeout()
    connector = aiohttp.TCPConnector(limit=max_concurrent_requests)

    total = len(user_prompts)
    if progress_title is not None:
        logging.info("Starting %s (%d requests)", progress_title, total)

    completed = 0

    async def _request(user_prompt: str):
        nonlocal completed
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
        if progress_title is not None:
            completed += 1
            logging.debug("%s: %d/%d done", progress_title, completed, total)
        return response

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        responses = await asyncio.gather(
            *[_request(user_prompt) for user_prompt in user_prompts]
        )

    if progress_title is not None:
        logging.info("Completed %s", progress_title)

    return responses
```

The `completed += 1` increment is safe under asyncio's single-threaded
cooperative model (no await between read and write). The `nonlocal` counter
tracks the running count of completions, which is what R4 specifies for the
DEBUG message (`<i>/<N>` where `<i>` is the running completed count).

The `from rich.progress import Progress` import is removed. No other file is
modified for output behavior.

### 2.2. Todo list

1. [ ] Add `import logging` to `rally/interaction.py`
2. [ ] Replace the rich `Progress` block in `_request_based_on_prompts` with logging-based progress (INFO start/finish, DEBUG per-request)
3. [ ] Remove the `from rich.progress import Progress` import from `rally/interaction.py`
4. [ ] Run linters: `black rally/`, `isort rally/`, `pylint rally/`, `mypy rally/`
5. [ ] Run the full test suite (`pytest`) and ensure all tests pass

### 2.3. Modification summary

| File | Action |
|------|--------|
| `rally/interaction.py` | Modified: add `import logging`; replace rich `Progress` block with `logging.info()`/`logging.debug()` based progress (INFO start/completion, DEBUG per-request); remove `from rich.progress import Progress` import. |
