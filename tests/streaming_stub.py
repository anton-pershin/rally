"""A scripted OpenAI-compatible streaming server for the streaming tests.

The hazards of a streaming client -- incremental delivery, a stalled read, a
connection dropped mid-stream, a rejected request -- cannot be exercised
against a mocked ``requests.post``, so these tests drive a real socket. The
responses use chunked transfer encoding, which is what makes an abruptly
closed connection distinguishable from a stream that ended cleanly.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


def content_line(text: str) -> str:
    """An SSE line whose chunk carries content."""
    return _event_line({"choices": [{"index": 0, "delta": {"content": text}}]})


def reasoning_line(text: str) -> str:
    """An SSE line whose chunk carries the reasoning trace, not content."""
    return _event_line(
        {"choices": [{"index": 0, "delta": {"reasoning_content": text}}]}
    )


def role_line() -> str:
    """An SSE line for the opening chunk, which carries no content."""
    return _event_line({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})


def usage_line(prompt_tokens: int, completion_tokens: int) -> str:
    """An SSE line for the usage report, which arrives without choices."""
    return _event_line(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    )


def done_line() -> str:
    """The provider's end-of-stream marker."""
    return "data: [DONE]\n\n"


def keepalive_line() -> str:
    """A comment line, as a middlebox sends to hold a connection open."""
    return ": keep-alive\n\n"


def broken_line() -> str:
    """A data line whose payload is not JSON."""
    return "data: {not json\n\n"


def _event_line(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        stub = self.server.stub  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            body: Any = json.loads(raw_body)
        except json.JSONDecodeError:
            body = {"raw_body": raw_body}
        stub.requests.append({"headers": dict(self.headers), "body": body})

        if stub.mode == "silent":
            time.sleep(stub.hold_seconds)
            return

        self.send_response(stub.status)
        if stub.status >= 300 or stub.mode == "json":
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(stub.body)))
            self.end_headers()
            self._write(stub.body)
            return

        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        if stub.mode == "silent_after_headers":
            time.sleep(stub.hold_seconds)
            return

        for index, line in enumerate(stub.lines):
            self._write_chunk(line)
            if stub.delay:
                time.sleep(stub.delay)
            if stub.mode == "abrupt" and index + 1 >= stub.abrupt_after:
                self._drop()
                return
        self._end_chunks()

    def _write_chunk(self, payload: str) -> None:
        data = payload.encode("utf-8")
        self._write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")

    def _end_chunks(self) -> None:
        self._write(b"0\r\n\r\n")

    def _write(self, data: bytes) -> None:
        """Write to the client, ignoring the disconnect of a client that stopped."""
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _drop(self) -> None:
        """Close the connection mid-response, without the terminating chunk."""
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
        self.close_connection = True


class _StubServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamingStub:
    """A scripted streaming server on an ephemeral port, used as a context manager.

    ``mode`` selects the behaviour: ``stream`` writes every line and ends the
    response properly, ``abrupt`` drops the connection after ``abrupt_after``
    lines, ``silent`` never answers, ``silent_after_headers`` sends the headers
    and then nothing, ``json`` answers with ``body`` as a plain JSON response.
    """

    def __init__(
        self,
        lines: Optional[list[str]] = None,
        delay: float = 0.0,
        mode: str = "stream",
        status: int = 200,
        body: bytes = b"",
        abrupt_after: int = 1,
        hold_seconds: float = 5.0,
    ) -> None:
        self.lines = list(lines) if lines is not None else []
        self.delay = delay
        self.mode = mode
        self.status = status
        self.body = body
        self.abrupt_after = abrupt_after
        self.hold_seconds = hold_seconds
        self.requests: list[dict[str, Any]] = []
        self._server: Optional[_StubServer] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StreamingStub":
        self._server = _StubServer(("127.0.0.1", 0), _Handler)
        self._server.stub = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("The stub server is not running.")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1/chat/completions"
