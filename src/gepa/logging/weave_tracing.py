# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Weave tracing utilities for hierarchical call tracing with feedback.

This module provides weave tracing integration. Weave is now a required dependency,
so we directly use weave.op for decorating functions. The call context propagation
works correctly because we use the native weave.op decorator.
"""

import weave

# Re-export weave.op directly - this ensures proper call context propagation
weave_op = weave.op


def add_call_feedback(
    score: float | None = None,
    scores: dict[str, float] | None = None,
) -> None:
    """Add feedback scores to the current weave call.

    This safely adds feedback scores to the current weave call.
    If no call is active, this is a no-op.

    Args:
        score: A single score value to add as feedback.
        scores: A dictionary of named scores to add as feedback.
    """
    try:
        call = weave.require_current_call()

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
