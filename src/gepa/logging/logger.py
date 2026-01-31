# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from __future__ import annotations

import os
import sys
from typing import Protocol

from rich.console import Console
from rich.panel import Panel


class LoggerProtocol(Protocol):
    def log(self, message: str): ...


# Color palette for header hashing
_HEADER_COLORS = ["cyan", "green", "yellow", "magenta", "blue", "red"]


def _color_for_header(header: str) -> str:
    """Return a consistent color for a given header based on its hash."""
    return _HEADER_COLORS[hash(header) % len(_HEADER_COLORS)]


class RichLogger(LoggerProtocol):
    """A logger that uses rich formatting with colorful output."""

    def __init__(self, console: Console | None = None, indent_level: int = 0):
        self.console = console or Console()
        self._debug_enabled = os.environ.get("LOG_LEVEL", "").upper() == "DEBUG"
        self._indent_level = indent_level

    def log(self, message: str, header: str | None = None, flush: bool = False):
        """Log a message with optional colored header.

        Args:
            message: The message to log.
            header: Optional header text that will be uppercased and colored.
            flush: Whether to force flush stdout after logging.
        """
        indent = "  " * self._indent_level
        if header:
            color = _color_for_header(header)
            formatted_header = f"[{color}][{header.upper()}][/{color}]"
            self.console.print(f"{indent}{formatted_header} {message}")
        else:
            self.console.print(f"{indent}{message}")
        if flush:
            sys.stdout.flush()

    def debug(self, message: str, header: str | None = None, flush: bool = True):
        """Log a debug message (only shown when LOG_LEVEL=DEBUG).

        Args:
            message: The message to log.
            header: Optional header text that will be uppercased and colored.
            flush: Whether to force flush stdout after logging (default True for debug).
        """
        if self._debug_enabled:
            self.log(message, header=header, flush=flush)

    def show(self, content: str, title: str | None = None):
        """Display content using a rich panel.

        Args:
            content: The content to display in the panel.
            title: Optional title for the panel.
        """
        indent = "  " * self._indent_level
        panel = Panel(content, title=title, expand=False)
        # Print indent separately then the panel
        if indent:
            self.console.print(indent, end="")
        self.console.print(panel)

    def indent(self) -> RichLogger:
        """Return a new logger with increased indentation level.

        Returns:
            A new RichLogger with indent_level incremented by 1.
        """
        return RichLogger(console=self.console, indent_level=self._indent_level + 1)


# Global default logger instance
_default_logger: RichLogger | None = None


def get_logger() -> RichLogger:
    """Get the global default RichLogger instance.

    Returns:
        The global RichLogger instance, creating one if needed.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = RichLogger()
    return _default_logger


class StdOutLogger(LoggerProtocol):
    def log(self, message: str):
        print(message)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            if hasattr(f, "flush"):
                f.flush()

    def isatty(self):
        # True if any of the files is a terminal
        return any(hasattr(f, "isatty") and f.isatty() for f in self.files)

    def close(self):
        for f in self.files:
            if hasattr(f, "close"):
                f.close()

    def fileno(self):
        for f in self.files:
            if hasattr(f, "fileno"):
                return f.fileno()
        raise OSError("No underlying file object with fileno")


class Logger(LoggerProtocol):
    def __init__(self, filename, mode="a"):
        self.file_handle = open(filename, mode)
        self.file_handle_stderr = open(filename.replace("run_log.", "run_log_stderr."), mode)
        self.modified_sys = False

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, self.file_handle)
        sys.stderr = Tee(sys.stderr, self.file_handle_stderr)
        self.modified_sys = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.file_handle.close()
        self.file_handle_stderr.close()
        self.modified_sys = False

    def log(self, *args, **kwargs):
        if self.modified_sys:
            print(*args, **kwargs)
        else:
            # Emulate print(*args, **kwargs) behavior but write to the file
            print(*args, **kwargs)
            print(*args, file=self.file_handle_stderr, **kwargs)
        self.file_handle.flush()
        self.file_handle_stderr.flush()
