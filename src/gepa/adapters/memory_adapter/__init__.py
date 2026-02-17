# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Memory Adapter for GEPA.

This adapter enables optimization of a memory store (key-value entries)
using edit-based mutation instead of full-text rewrite.

Exports:
    MemoryAdapter: Main adapter class
    MemoryDataInst: Dataset item type
    MemoryTrajectory: Execution trace type
    MemoryOutput: Output type
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_adapter import MemoryAdapter, MemoryDataInst, MemoryOutput, MemoryTrajectory
    from .routing_adapter import RoutingMemoryAdapter

__all__ = [
    "MemoryAdapter",
    "MemoryDataInst",
    "MemoryOutput",
    "MemoryTrajectory",
    "RoutingMemoryAdapter",
]


def __getattr__(name: str):
    """Lazy import to handle missing dependencies gracefully."""
    if name in {"MemoryAdapter", "MemoryDataInst", "MemoryOutput", "MemoryTrajectory"}:
        from .memory_adapter import MemoryAdapter, MemoryDataInst, MemoryOutput, MemoryTrajectory

        return locals()[name]

    if name == "RoutingMemoryAdapter":
        from .routing_adapter import RoutingMemoryAdapter

        return RoutingMemoryAdapter

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
