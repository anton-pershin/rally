## 04-streaming-kiss-spec

### 1. Requirement analysis

#### 1.1 Motivation

rally owns what an LLM request is: `Llm` builds its own headers and body, and the operations that send them live on it (spec 03). Streaming is the one request shape rally cannot send, and the only consumer that needs it is slam-eval's performance-monitoring path — with monitoring on, the collector performs the generation itself and must observe the server's chunks as they arrive, because its metrics are timing: TTFT from the first content chunk, TPOT from the intervals between them. Today that consumer therefore owns a client of its own: SSE framing, `delta`, the usage chunk, the end marker, and the mapping of every failure mode. All of it is provider knowledge, and provider knowledge is rally's. This spec gives rally the streaming operation, so a consumer can ask for a stream and measure it without speaking the provider's dialect. Re-pointing that consumer is its own spec on the slam side.

#### 1.2 Functional requirements

**FR1.** `Llm` gains a synchronous streaming operation, `stream(message_history)`, sending the `Llm`'s own headers and its own body (`build_headers()`, `build_payload()`) plus exactly `stream: true` and `stream_options: {"include_usage": true}`.

**FR2.** The operation yields typed stream events, not raw provider chunks. An event carries the content delta of its chunk (or none), the reasoning delta when the server streams the reasoning trace separately (or none), the usage report when the server sends one (or none), and whether it is the final event. A consumer never reads provider field names — `choices`, `delta`, the usage key, the end marker — to understand an event.

**FR3.** Delivery is incremental: each event is yielded as its chunk arrives, so a consumer can timestamp it, and events are in arrival order. The operation does not buffer the response and return it at the end — TTFT and TPOT are measurable only from arrival times.

**FR4.** The end of the stream is visible to the consumer as an explicit final event; no provider marker has to be decoded to learn that the stream is over.

**FR5.** A consumer can assemble the completion by concatenating event contents, and can know how many events carried content, so a server that reports no usage still leaves the consumer able to count content-bearing events.

**FR6.** The reasoning trace is delivered, not dropped: when a server streams it separately from the content, the reasoning text is visible on the event and is never mixed into the content.

**FR7.** Failures are distinguishable by the consumer, each with its own type: authorization failure (HTTP 401 or 403), a rejected streaming request (another HTTP status, with the status available to the consumer), a transport failure (connection refused, reset, or dropped mid-stream), and a read that exceeds the timeout.

**FR8.** The streamed request is the same request description as the non-streaming one: endpoint, headers and body come from the same builders, so every generation parameter configured on the `Llm` — `model`, the output-token cap, `enable_thinking` — applies to a stream unchanged, with no consumer-side request shaping.

**FR9.** A timeout bounds the streaming read, is configurable on the `Llm`, and its expiry surfaces as the timeout failure of FR7.

**FR10.** The existing non-streaming operations (`request`, `arequest`, `request_batch`, `arequest_batch`) keep their exact semantics, including returning `None` on an unsuccessful response. No consumer of those operations is edited.

**FR11.** A consumer that stops iterating early releases the response: nothing hangs and nothing leaks.

**FR12.** Tests cover the wire (both streaming keys sent, headers and body taken from the builders), incremental delivery, the usage report, the reasoning channel, the end of stream, and every failure of FR7.

#### 1.3 Non-functional requirements

**NFR1.** No new dependency: `requests` already streams a response; rally's dependency list is unchanged.

**NFR2.** The sync/async split is unchanged. The streaming operation is synchronous and uses `requests`; no event loop is introduced into the synchronous path, and this spec adds no asynchronous variant.

**NFR3.** rally's linters report no message absent from the pre-change `main` — `pylint` and `isort` are not clean there, so the gate is the difference.

**NFR4.** Blast radius: nothing outside rally changes. slam-core, slam-eval's collector, kygs, entity-processing and lecture-me are re-pointed by their own follow-up specs, and no consumer call site is touched here.

**NFR5.** Provider knowledge stays inside rally: no consumer parses SSE framing, chunk deltas, usage placement or the end marker to use this operation.

#### 1.4 Expected behavioural variants

| # | Situation | Expected behaviour |
|---|-----------|--------------------|
| 1 | a completion streamed with content chunks | each content chunk is delivered as its own event, in arrival order |
| 2 | the server reports usage in a final chunk | the usage report is visible to the consumer, with prompt and completion token counts |
| 3 | the server streams the reasoning trace separately | the reasoning text is visible on its events and never mixed into the content |
| 4 | the server reports no usage | the consumer can count content-bearing events instead |
| 5 | a chunk carries no content (role-only or empty delta) | delivered as an event with no content, not counted as content |
| 6 | the server sends the end marker | an explicit final event is delivered and iteration ends normally |
| 7 | the stream ends without an end marker | iteration ends at the close of the response, the final event still delivered |
| 8 | HTTP 401 or 403 | the authorization failure of FR7 is raised |
| 9 | another HTTP status | the rejection of FR7 is raised, carrying the status |
| 10 | connection refused or reset before any content | the transport failure of FR7 is raised |
| 11 | the connection drops mid-stream | the transport failure of FR7 is raised, after the events already delivered |
| 12 | no content arrives at all | no content event and no exception: emptiness is the consumer's inference, not a failure |
| 13 | the read exceeds the timeout | the timeout failure of FR7 is raised |
| 14 | a cap is configured on the `Llm` | the streamed body carries it, exactly as the non-streaming body does |
| 15 | thinking is configured on the `Llm` | the streamed body carries `chat_template_kwargs`, exactly as the non-streaming body does |
| 16 | no authorization is configured | no `Authorization` header is sent |
| 17 | a consumer stops iterating early | the response is released; nothing hangs and nothing leaks |
| 18 | an existing non-streaming operation is called | unchanged behaviour and returns |

### 2. Tests

Test infrastructure: a stub streaming server, added as the helper `tests/streaming_stub.py` (standard library `http.server` in a thread — no new dependency). It serves a scripted SSE response: a list of payloads, an optional delay between them, an optional status code, an optional abrupt close, an optional never-respond mode; it listens on an ephemeral port and records the headers and body of the request it received. Incremental delivery, the timeout and the transport failures cannot be tested against a mocked `requests.post`, and the recorded request is the wire evidence for the literal body assertions. Every test below is synchronous — this spec adds no asynchronous variant (NFR2) — so nothing is marked for `anyio`.

#### 2.1 The streamed request (FR1, FR8 — rows 14, 15, 16, 19) — `tests/test_llm.py`

- **T1** — `TestStream::test_stream_posts_built_headers_and_body_with_stream_keys`: the stub receives the `Llm`'s headers and a body equal to the literal expected dict — the `Llm`'s configured fields plus `"stream": true` and `"stream_options": {"include_usage": true}` — asserted literally, not against `build_headers()`/`build_payload()`, so the test is independent of the builder (FR1, row 19).
- **T2** — `TestStream::test_stream_body_carries_generation_parameters`: a cap configured on the `Llm` arrives as both `max_completion_tokens` and `max_tokens`, `enable_thinking` arrives as `chat_template_kwargs`, and `model` arrives with its value — exactly as the non-streaming body carries them (FR8, rows 14, 15).
- **T3** — `TestStream::test_stream_without_authorization_sends_no_authorization_header`: with `authorization=None` the received headers carry no `Authorization` key; with a value they carry it verbatim (row 16).

#### 2.2 Event delivery (FR2–FR6 — rows 1–7, 20) — `tests/test_llm.py`

- **T4** — `TestStreamEvents::test_content_chunks_yield_one_event_each_in_order`: three content chunks produce one event per chunk, their contents in arrival order, and their concatenation equals the completion text (FR2, FR5, row 1).
- **T5** — `TestStreamEvents::test_usage_is_visible_with_both_token_counts`: a final usage chunk yields an event carrying the prompt and completion token counts (FR2, row 2).
- **T6** — `TestStreamEvents::test_absent_usage_leaves_usage_unset_and_content_events_countable`: with no usage chunk every event's usage is unset and the number of content-bearing events is observable to the consumer (FR5, row 4).
- **T7** — `TestStreamEvents::test_reasoning_is_separate_from_content`: a chunk carrying reasoning text yields an event whose reasoning is set and whose content is unset, and the reasoning text appears in no concatenated content (FR6, row 3).
- **T8** — `TestStreamEvents::test_contentless_chunk_yields_event_without_content`: a role-only or empty-delta chunk yields an event carrying no content, and it does not count as a content-bearing event (FR2, row 5).
- **T9** — `TestStreamEvents::test_end_marker_yields_final_event_and_closes_iteration`: the end marker produces a final event and the iteration then stops (FR4, row 6).
- **T10** — `TestStreamEvents::test_stream_without_end_marker_ends_at_response_close`: a response that simply closes, with no marker, still ends the iteration, the final event delivered (FR4, row 7).
- **T11** — `TestStreamEvents::test_events_arrive_incrementally_not_buffered`: the stub sends its chunks with a known delay between them; the consumer records the wall-clock time of each event as it iterates, and the gaps between the first event and the later ones reflect those delays — a buffering implementation fails this (FR3, row 1). The load-bearing test for TTFT and TPOT: without it, reading the whole response and then yielding is indistinguishable from a correct implementation.
- **T12** — `TestStreamEvents::test_non_chunk_lines_are_skipped`: a keep-alive or malformed line among valid chunks does not abort the stream, and the surrounding content is delivered (FR2, row 20).

#### 2.3 Failures and the timeout (FR7, FR9 — rows 8–13) — `tests/test_llm.py`

- **T13** — `TestStreamFailures::test_authorization_failure_is_its_own_error`: HTTP 401 and HTTP 403 each raise the authorization error (row 8).
- **T14** — `TestStreamFailures::test_rejected_stream_raises_with_status`: another status (say 500, with a JSON error body) raises the rejection error, and the status is readable on it (row 9).
- **T15** — `TestStreamFailures::test_transport_failure_before_content`: an `Llm` pointed at a closed port raises the transport error (row 10).
- **T16** — `TestStreamFailures::test_transport_failure_mid_stream_after_delivered_events`: the stub sends one content chunk and then closes the connection abruptly; the events already delivered were yielded, and the transport error follows (row 11).
- **T17** — `TestStreamFailures::test_stream_with_no_content_is_not_an_error`: the end marker arrives with no content chunk at all — no exception, and the consumer sees zero content-bearing events (row 12).
- **T18** — `TestStreamFailures::test_timeout_is_its_own_error_and_respects_the_configured_value`: with a small timeout configured on the `Llm` and a stub that accepts the connection and never sends, the timeout error is raised — the timeout type, not merely the transport type (FR9, row 13).
- **T19** — `TestStreamFailures::test_failure_types_are_distinguishable`: one place asserting the four types are mutually distinguishable (authorization, rejection, transport, timeout), including that a plain transport failure is not the timeout type, so a consumer can branch on them (FR7).

#### 2.4 Early stop, the non-streaming surface, and the suite (FR10, FR11 — rows 17, 18; NFR1–NFR5)

- **T20** — `TestStreamLifecycle::test_early_stop_releases_the_response`: a consumer that iterates two events and breaks leaves nothing open — the response is closed (asserted on the response object, or on the stub observing the disconnect) — and the test does not hang (FR11, row 17).
- **T21** — `TestNonStreamingUnchanged::test_existing_operations_keep_their_semantics`: the existing `tests/test_llm.py` cases for `request`, `arequest`, `request_batch` and `arequest_batch` pass untouched, and none of them raises any of the new error types (FR10, row 18).
- **T22** — the suite (NFR1–NFR5): `pytest` over `rally/tests/` — the 60 existing tests plus the new ones — reports no failures; `black --check`, `isort --check` and `mypy` are clean over `rally/`; `pylint rally/` reports no message absent from `main`, compared between a clean `main` clone and a branch clone, because the ignored `/hydra/` job-output directory otherwise makes `isort` and `pylint` classify `hydra` as first-party and changes both verdicts. The `requests.post` missing-timeout message on the existing non-streaming path is pre-existing and stays; the new streaming call site passes a timeout (NFR3).
- The wire claims are asserted by the literal expectations of T1–T3 rather than against the builders (NFR5), and no consumer repo is exercised here: `slam-core`, `slam-eval`, `kygs`, `entity-processing` and `lecture-me` are untouched by this spec and re-pointed by their own follow-ups (NFR4). No manual smoke test is needed — the stub server gives real wire evidence, which a mocked transport cannot.

### 3. Implementation plan

#### 3.1 Implementation repos

[List all the repos expected to be involved in any implementation]

#### 3.2 High-level design

[Mermaid diagram (typically, flowchart, but it is ultimately up to a planner) describing the high-level design. Take the high-level design from the constitution/validation spec and draw how your proposal fits into it.]

#### 3.3 Todo list

[Write a todo list with all the steps necessary to create an implementation which will allow the tests to be passed. Below is the template where the first two steps are mandatory]

1. [ ] Write the tests
2. [ ] Run all the tests and ensure that they fail
3. [ ] ...

#### 3.4 Modification summary

[Fill the table below specifying which files are going to be modified and which are going to be created]

| File | Action |
|------|--------|
| ... | Modified: add X, modify Y, etc. |
| ... | New |
