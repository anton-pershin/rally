## Terminology

### General

| Term | Definition |
|------|------------|
| LLM endpoint | The configured address used to send requests to a local or external LLM. |
| Synchronous request | A request that blocks its caller until it gets a response or fails. |
| Asynchronous request | A request that lets its caller perform other work while it waits for a response. |
| Batch request | A request operation that processes multiple prompts. |
| Streaming | Incremental delivery of a response as the LLM endpoint produces it. |

### Presentation layer

| Term | Definition |
|------|------------|
| Chat script | The executable script that loads configuration, interacts with the user, and calls the configured LLM. |

### Application layer

| Term | Definition |
|------|------------|
| Message history | An ordered list of messages sent to an LLM endpoint. |
| Reasoning removal function | A model-family-specific function that removes reasoning text from a response. |

### Infrastructure layer

| Term | Definition |
|------|------------|
| HTTP client | A library that sends HTTP requests to an LLM endpoint. |
