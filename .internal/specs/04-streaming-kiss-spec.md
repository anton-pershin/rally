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

[List all the tests explicitly covering all the requirements and the expected variants mentioned in the requirement analysis. Tag them as T1, T2 etc.]

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
