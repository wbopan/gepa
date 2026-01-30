# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Weave tracing utilities for hierarchical call tracing with feedback."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    import weave as weave_module

# Global state for weave tracing
_weave_enabled: bool = False
_weave_client: weave_module | None = None

F = TypeVar("F", bound=Callable[..., Any])


def configure_weave_tracing(enabled: bool, client: weave_module | None = None) -> None:
    """Configure the global weave tracing state.

    Args:
        enabled: Whether weave tracing is enabled.
        client: The weave module (import weave) when enabled, None otherwise.
    """
    global _weave_enabled, _weave_client
    _weave_enabled = enabled
    _weave_client = client


def is_weave_tracing_enabled() -> bool:
    """Check if weave tracing is currently enabled."""
    return _weave_enabled


def weave_op(name: str | None = None) -> Callable[[F], F]:
    """Conditional weave.op decorator that becomes a no-op when tracing is disabled.

    Args:
        name: Optional operation name for weave tracing.

    Returns:
        A decorator that wraps the function with weave.op when enabled,
        or returns the original function when disabled.
    """

    def decorator(fn: F) -> F:
        # Store the original function and name for later wrapping
        # We need to check _weave_enabled at call time, not decoration time
        original_fn = fn
        op_name = name

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _weave_enabled or _weave_client is None:
                return original_fn(*args, **kwargs)

            # Create and apply weave.op at call time
            weave_decorated = _weave_client.op(name=op_name)(original_fn) if op_name else _weave_client.op()(original_fn)
            return weave_decorated(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def add_call_feedback(
    score: float | None = None,
    scores: dict[str, float] | None = None,
) -> None:
    """Add feedback scores to the current weave call.

    This safely adds feedback scores to the current weave call.
    If weave tracing is disabled or no call is active, this is a no-op.

    Args:
        score: A single score value to add as feedback.
        scores: A dictionary of named scores to add as feedback.
    """
    if not _weave_enabled or _weave_client is None:
        return

    try:
        call = _weave_client.require_current_call()

        # Add single score feedback
        if score is not None:
            call.feedback.add("score", {"value": score})

        # Add multiple named scores
        if scores is not None:
            for name, value in scores.items():
                call.feedback.add(name, {"value": value})

    except Exception:
        # Silently ignore if no current call or other errors
        pass
