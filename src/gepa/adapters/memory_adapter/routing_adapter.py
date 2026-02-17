# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Routing Memory Adapter — LLM-based memory routing for GEPA.

Extends MemoryAdapter with a two-step evaluation flow:
1. An LLM selects the most relevant memory entry keys for each query.
2. Only the selected entries are loaded into context for answer generation.

This enables efficient caching: when a mutation only changes one entry's content,
queries routed to other entries produce identical LLM calls and hit the litellm cache.
"""

import re
from typing import Callable

from gepa.logging import get_logger

from .memory_adapter import MemoryAdapter, MemoryDataInst
from .memory_store import MemoryEntry, format_memory_as_markdown, parse_memory_xml

logger = get_logger()

KEY_SELECTION_PROMPT = """\
Select up to {top_k} of the most relevant knowledge entries for answering the query below.

Available entries:
{entry_list}

Query: {query}

Output ONLY the entry numbers, comma-separated. Do not include any explanation. Example: 1, 3"""


class RoutingMemoryAdapter(MemoryAdapter):
    """MemoryAdapter with LLM-based routing — selects relevant entries per query.

    Instead of injecting all memory entries into every LLM call, this adapter
    first asks a (cheap, cached) routing LLM to pick the top-k entries per query,
    then builds per-item system prompts containing only the selected entries.

    Routing is deterministic (temperature=0) — this is a correctness requirement,
    not just an optimization. The GEPA EvaluationCache keys on (candidate_hash,
    example_id), and deterministic routing ensures the same candidate + query
    always produces the same routed context and thus the same score.

    Args:
        routing_model: Model for key selection (litellm model name or callable).
            Defaults to task_model.
        route_top_k: Maximum number of entries to select per query.
        **kwargs: Forwarded to MemoryAdapter.__init__.
    """

    def __init__(
        self,
        *,
        routing_model: str | Callable | None = None,
        route_top_k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.routing_model = routing_model if routing_model is not None else self.task_model
        # Ensure litellm is imported if routing_model is a string
        if isinstance(self.routing_model, str) and not hasattr(self, "_litellm"):
            import litellm

            self._litellm = litellm
        self.route_top_k = route_top_k

    def _build_batch_contexts(
        self,
        batch: list[MemoryDataInst],
        memory_xml: str,
    ) -> list[tuple[str, str]]:
        """Override: route each query to top-k entries via LLM."""
        try:
            entries = parse_memory_xml(memory_xml)
        except ValueError:
            entries = []

        # If entries fit within top_k, no routing needed — use all entries
        if len(entries) <= self.route_top_k:
            return super()._build_batch_contexts(batch, memory_xml)

        queries = [item["input"] for item in batch]
        per_item_indices = self._route_batch(queries, entries)

        contexts: list[tuple[str, str]] = []
        for selected_indices in per_item_indices:
            selected_entries = [entries[i] for i in selected_indices]
            md = format_memory_as_markdown(selected_entries)
            sp = self._format_system_prompt(md)
            contexts.append((sp, md))
        return contexts

    def _route_batch(
        self,
        queries: list[str],
        entries: list[MemoryEntry],
    ) -> list[list[int]]:
        """Batch LLM key-selection calls.

        Args:
            queries: User queries to route.
            entries: Parsed MemoryEntry objects.

        Returns:
            List of 0-indexed entry index lists, one per query.
        """
        entry_list = "\n".join(f"{i + 1}. {e.key}" for i, e in enumerate(entries))

        messages_list = [
            [
                {
                    "role": "user",
                    "content": KEY_SELECTION_PROMPT.format(
                        top_k=self.route_top_k,
                        entry_list=entry_list,
                        query=query,
                    ),
                }
            ]
            for query in queries
        ]

        logger.debug(
            f"Routing {len(queries)} queries across {len(entries)} entries (top_k={self.route_top_k})",
            header="route",
        )

        # Call routing model
        responses: list[str] = []
        if isinstance(self.routing_model, str):
            try:
                batch_responses = self._litellm.batch_completion(
                    model=self.routing_model,
                    messages=messages_list,
                    temperature=0,
                    num_retries=5,
                    caching=True,
                )
                for resp in batch_responses:
                    if isinstance(resp, Exception):
                        raise resp
                    content = resp.choices[0].message.content  # type: ignore[union-attr]
                    responses.append(content.strip() if content else "")
            except Exception as e:
                logger.log(f"Routing batch completion failed: {e}", header="error")
                # Fallback: select first top_k entries for all queries
                fallback = list(range(min(self.route_top_k, len(entries))))
                return [fallback] * len(queries)
        else:
            for msgs in messages_list:
                try:
                    result = self.routing_model(msgs)
                    responses.append(result if isinstance(result, str) else str(result))
                except Exception as e:
                    logger.log(f"Routing model call failed: {e}", header="error")
                    responses.append("")

        # Parse responses into index lists
        per_item_indices: list[list[int]] = []
        for resp_text in responses:
            indices = self._parse_index_selection(resp_text, len(entries))
            if not indices:
                # Fallback: first top_k entries
                indices = list(range(min(self.route_top_k, len(entries))))
                logger.debug(f"Routing parse returned empty, falling back to first {len(indices)} entries", header="route")
            # Enforce top_k cap — LLM may return more than requested
            indices = indices[: self.route_top_k]
            logger.debug(f"Routed to entries: {indices} from response: {resp_text!r}", header="route")
            per_item_indices.append(indices)

        return per_item_indices

    @staticmethod
    def _parse_index_selection(text: str, num_entries: int) -> list[int]:
        """Parse comma-separated 1-indexed numbers into 0-indexed entry indices.

        Handles: "1, 3, 5", "1,3,5", "1\\n3\\n5", and mixed formats.
        Ignores out-of-range numbers and non-numeric tokens.

        Args:
            text: Raw LLM output text.
            num_entries: Total number of available entries.

        Returns:
            List of unique 0-indexed entry indices in order of appearance.
        """
        indices: list[int] = []
        for token in re.split(r"[,\n]", text):
            token = token.strip().rstrip(".")
            match = re.match(r"(\d+)", token)
            if match:
                idx = int(match.group(1)) - 1  # 1-indexed → 0-indexed
                if 0 <= idx < num_entries and idx not in indices:
                    indices.append(idx)
        return indices
