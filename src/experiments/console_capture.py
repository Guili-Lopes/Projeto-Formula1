"""Capture legacy print output while preserving terminal visibility."""

from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class _TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def capture_console(log_path: Path, *, echo: bool = True) -> Iterator[None]:
    """Redirect stdout and stderr to a UTF-8 log and optionally the terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        stdout_stream: TextIO = _TeeStream(sys.stdout, handle) if echo else handle
        stderr_stream: TextIO = _TeeStream(sys.stderr, handle) if echo else handle
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            yield
