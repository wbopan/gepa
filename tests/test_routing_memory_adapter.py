# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Tests for the Routing Memory Adapter.

Tests cover:
- _parse_index_selection: valid, partial match, out-of-range, verbose model output, garbage
- _build_batch_contexts with mock routing LLM
- Fallback to parent when entries <= top_k
- Per-item system prompts contain only selected entries' content
- End-to-end evaluate() with routing
"""

import pytest

from gepa.adapters.memory_adapter.memory_adapter import MemoryAdapter, MemoryDataInst
from gepa.adapters.memory_adapter.routing_adapter import RoutingMemoryAdapter

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_evaluator():
    """Simple evaluator that checks if answer is in response."""

    def evaluator(data: MemoryDataInst, response: str):
        answer = data.get("answer", "")
        if answer and answer.lower() in response.lower():
            return {"score": 1.0, "feedback": "Correct answer found in response."}
        return {"score": 0.0, "feedback": f"Expected '{answer}' not found in response."}

    return evaluator


@pytest.fixture
def mock_task_model():
    """Mock task model callable."""

    def model(messages):
        return "The answer is 42."

    return model


@pytest.fixture
def mock_routing_model():
    """Mock routing model that always selects entries 1 and 3 (1-indexed)."""

    def model(messages):
        return "1, 3"

    return model


@pytest.fixture
def populated_memory_xml():
    """Memory XML with 5 entries for routing tests."""
    return (
        "<memory>\n"
        '<entry key="API Authentication">Use Bearer tokens for auth</entry>\n'
        '<entry key="Error Handling">Retry on 5xx errors</entry>\n'
        '<entry key="Date Formatting">Use ISO 8601 format</entry>\n'
        '<entry key="Rate Limiting">Max 100 requests per minute</entry>\n'
        '<entry key="Logging">Use structured JSON logging</entry>\n'
        "</memory>"
    )


@pytest.fixture
def populated_candidate(populated_memory_xml):
    """Candidate dict with 5 memory entries."""
    return {"memory": populated_memory_xml}


@pytest.fixture
def routing_adapter(simple_evaluator, mock_task_model, mock_routing_model):
    """Create a RoutingMemoryAdapter with mock models."""
    return RoutingMemoryAdapter(
        task_model=mock_task_model,
        reflection_model=mock_task_model,
        evaluator=simple_evaluator,
        routing_model=mock_routing_model,
        route_top_k=2,
        base_system_prompt="You are a helpful assistant.",
    )


# ============================================================================
# _parse_index_selection tests
# ============================================================================


class TestParseIndexSelection:
    """Tests for _parse_index_selection."""

    def test_comma_separated(self):
        result = RoutingMemoryAdapter._parse_index_selection("1, 3, 5", 5)
        assert result == [0, 2, 4]

    def test_no_spaces(self):
        result = RoutingMemoryAdapter._parse_index_selection("1,3,5", 5)
        assert result == [0, 2, 4]

    def test_newline_separated(self):
        result = RoutingMemoryAdapter._parse_index_selection("1\n3\n5", 5)
        assert result == [0, 2, 4]

    def test_mixed_separators(self):
        result = RoutingMemoryAdapter._parse_index_selection("1, 3\n5", 5)
        assert result == [0, 2, 4]

    def test_out_of_range_ignored(self):
        result = RoutingMemoryAdapter._parse_index_selection("1, 3, 10", 5)
        assert result == [0, 2]

    def test_zero_ignored(self):
        """0 is out of range for 1-indexed input."""
        result = RoutingMemoryAdapter._parse_index_selection("0, 1, 2", 5)
        assert result == [0, 1]

    def test_verbose_model_output(self):
        """Model outputs entry numbers with labels."""
        text = "1. API Authentication\n3. Date Formatting\n5. Logging"
        result = RoutingMemoryAdapter._parse_index_selection(text, 5)
        assert result == [0, 2, 4]

    def test_garbage_input(self):
        result = RoutingMemoryAdapter._parse_index_selection("no numbers here", 5)
        assert result == []

    def test_duplicates_removed(self):
        result = RoutingMemoryAdapter._parse_index_selection("1, 1, 3, 3", 5)
        assert result == [0, 2]

    def test_empty_string(self):
        result = RoutingMemoryAdapter._parse_index_selection("", 5)
        assert result == []

    def test_trailing_dots(self):
        result = RoutingMemoryAdapter._parse_index_selection("1., 3.", 5)
        assert result == [0, 2]

    def test_negative_ignored(self):
        """Negative numbers should not produce valid indices."""
        result = RoutingMemoryAdapter._parse_index_selection("-1, 2", 5)
        assert result == [1]


# ============================================================================
# _build_batch_contexts tests
# ============================================================================


class TestBuildBatchContexts:
    """Tests for _build_batch_contexts with routing."""

    def test_routing_selects_correct_entries(self, routing_adapter, populated_candidate):
        """Routing model returns '1, 3' → entries 0 and 2 (API Authentication, Date Formatting)."""
        batch: list[MemoryDataInst] = [
            {"input": "How do I authenticate?", "answer": "Bearer tokens"},
        ]
        memory_xml = populated_candidate["memory"]
        contexts = routing_adapter._build_batch_contexts(batch, memory_xml)

        assert len(contexts) == 1
        system_prompt, _memory_md = contexts[0]

        # Should contain selected entries
        assert "API Authentication" in system_prompt
        assert "Date Formatting" in system_prompt

        # Should NOT contain non-selected entries
        assert "Error Handling" not in system_prompt
        assert "Rate Limiting" not in system_prompt
        assert "Logging" not in system_prompt

    def test_fallback_when_few_entries(self, routing_adapter):
        """When entries <= top_k, routing is skipped and all entries are used."""
        # Only 2 entries, top_k=2 → no routing needed
        memory_xml = '<memory>\n<entry key="A">Content A</entry>\n<entry key="B">Content B</entry>\n</memory>'
        batch: list[MemoryDataInst] = [{"input": "test", "answer": "test"}]
        contexts = routing_adapter._build_batch_contexts(batch, memory_xml)

        assert len(contexts) == 1
        system_prompt, _ = contexts[0]
        assert "Content A" in system_prompt
        assert "Content B" in system_prompt

    def test_per_item_routing(self, simple_evaluator, mock_task_model):
        """Each query gets its own routing result."""
        call_count = 0

        def per_query_routing_model(messages):
            nonlocal call_count
            call_count += 1
            # First query routes to entry 1, second to entry 2
            if call_count == 1:
                return "1"
            return "2"

        adapter = RoutingMemoryAdapter(
            task_model=mock_task_model,
            reflection_model=mock_task_model,
            evaluator=simple_evaluator,
            routing_model=per_query_routing_model,
            route_top_k=1,
            base_system_prompt="Base prompt.",
        )

        memory_xml = (
            "<memory>\n"
            '<entry key="Alpha">Alpha content</entry>\n'
            '<entry key="Beta">Beta content</entry>\n'
            '<entry key="Gamma">Gamma content</entry>\n'
            "</memory>"
        )
        batch: list[MemoryDataInst] = [
            {"input": "Query about alpha", "answer": "alpha"},
            {"input": "Query about beta", "answer": "beta"},
        ]

        contexts = adapter._build_batch_contexts(batch, memory_xml)
        assert len(contexts) == 2

        # First query → entry 0 (Alpha)
        assert "Alpha content" in contexts[0][0]
        assert "Beta content" not in contexts[0][0]

        # Second query → entry 1 (Beta)
        assert "Beta content" in contexts[1][0]
        assert "Alpha content" not in contexts[1][0]


# ============================================================================
# End-to-end evaluate() tests
# ============================================================================


class TestRoutingEvaluate:
    """Tests for evaluate() with routing."""

    def test_evaluate_with_routing(self, routing_adapter, populated_candidate):
        batch: list[MemoryDataInst] = [
            {"input": "How do I authenticate?", "answer": "42"},
            {"input": "What about rate limits?", "answer": "blue"},
        ]
        result = routing_adapter.evaluate(batch, populated_candidate)
        assert len(result.outputs) == 2
        assert len(result.scores) == 2
        assert result.trajectories is None

    def test_evaluate_with_traces_and_routing(self, routing_adapter, populated_candidate):
        batch: list[MemoryDataInst] = [
            {"input": "How do I authenticate?", "answer": "42"},
        ]
        result = routing_adapter.evaluate(batch, populated_candidate, capture_traces=True)
        assert result.trajectories is not None
        assert len(result.trajectories) == 1

        traj = result.trajectories[0]
        # Trajectory should have the routed system prompt (not full memory)
        assert "API Authentication" in traj["system_prompt"]
        assert "Date Formatting" in traj["system_prompt"]
        # Non-selected entries should be absent
        assert "Rate Limiting" not in traj["system_prompt"]

    def test_evaluate_empty_memory_with_routing(self, routing_adapter):
        """Empty memory should work without routing."""
        batch: list[MemoryDataInst] = [{"input": "test", "answer": "42"}]
        candidate = {"memory": "<memory>\n</memory>"}
        result = routing_adapter.evaluate(batch, candidate)
        assert len(result.outputs) == 1


# ============================================================================
# Constructor tests
# ============================================================================


class TestRoutingAdapterInit:
    """Tests for RoutingMemoryAdapter constructor."""

    def test_routing_model_defaults_to_task_model(self, simple_evaluator):
        """When routing_model is not specified, it should default to task_model."""

        def task_model(messages):
            return "response"

        adapter = RoutingMemoryAdapter(
            task_model=task_model,
            reflection_model=task_model,
            evaluator=simple_evaluator,
            route_top_k=3,
        )
        # When task_model is callable, routing_model should also be that callable
        assert adapter.routing_model is task_model

    def test_explicit_routing_model(self, simple_evaluator, mock_task_model):
        """Explicit routing_model should be used."""

        def custom_router(messages):
            return "1"

        adapter = RoutingMemoryAdapter(
            task_model=mock_task_model,
            reflection_model=mock_task_model,
            evaluator=simple_evaluator,
            routing_model=custom_router,
            route_top_k=5,
        )
        assert adapter.routing_model is custom_router
        assert adapter.route_top_k == 5


# ============================================================================
# Reflective dataset and routing info tests
# ============================================================================


class TestReflectiveDatasetWithRouting:
    """Tests for make_reflective_dataset() with per-query routing info."""

    def test_make_reflective_dataset_includes_entries_used(self, routing_adapter, populated_candidate):
        """Reflective dataset examples should include 'Entries Used' with correct entry keys."""
        batch: list[MemoryDataInst] = [
            {"input": "How do I authenticate?", "answer": "Bearer tokens"},
            {"input": "What about rate limits?", "answer": "100 requests"},
        ]
        # Routing mock returns "1, 3" → entries "API Authentication" and "Date Formatting"
        eval_result = routing_adapter.evaluate(batch, populated_candidate, capture_traces=True)
        reflective_data = routing_adapter.make_reflective_dataset(populated_candidate, eval_result, ["memory"])

        assert "memory" in reflective_data
        examples = reflective_data["memory"]
        assert len(examples) == 2

        for ex in examples:
            assert "Entries Used" in ex
            assert ex["Entries Used"] == ["API Authentication", "Date Formatting"]

    def test_format_feedback_examples_with_entries_used(self):
        """_format_feedback_examples should include **Entries Used:** line."""
        examples = [
            {
                "Inputs": {"user_input": "How to auth?", "expected_answer": "Use tokens"},
                "Entries Used": ["API Authentication", "Rate Limiting"],
                "Generated Outputs": "Use Bearer tokens.",
                "Feedback": "Correct.",
            },
            {
                "Inputs": {"user_input": "Date format?", "expected_answer": "ISO 8601"},
                "Entries Used": ["Date Formatting"],
                "Generated Outputs": "Use ISO 8601.",
                "Feedback": "Correct.",
            },
        ]
        result = RoutingMemoryAdapter._format_feedback_examples(examples)

        assert "**Entries Used:** API Authentication, Rate Limiting" in result
        assert "**Entries Used:** Date Formatting" in result
        # Verify ordering: Entries Used comes after User Input and before Expected Answer
        lines = result.split("\n")
        for i, line in enumerate(lines):
            if "**Entries Used:**" in line:
                # Previous non-empty line should be User Input
                prev = lines[i - 1]
                assert "**User Input:**" in prev
                # Next line should be Expected Answer
                nxt = lines[i + 1]
                assert "**Expected Answer:**" in nxt

    def test_format_feedback_examples_without_entries_used(self):
        """_format_feedback_examples should skip **Entries Used:** line when key is absent."""
        examples = [
            {
                "Inputs": {"user_input": "test", "expected_answer": "answer"},
                "Generated Outputs": "response",
                "Feedback": "ok",
            },
        ]
        result = RoutingMemoryAdapter._format_feedback_examples(examples)
        assert "**Entries Used:**" not in result
        assert "**User Input:** test" in result

    def test_extract_keys_from_markdown(self):
        """_extract_keys_from_markdown should parse ## headers."""
        md = "## API Authentication\nUse Bearer tokens\n\n## Rate Limiting\nMax 100 req/min"
        keys = RoutingMemoryAdapter._extract_keys_from_markdown(md)
        assert keys == ["API Authentication", "Rate Limiting"]

    def test_extract_keys_from_empty_markdown(self):
        """Empty markdown should return empty list."""
        assert RoutingMemoryAdapter._extract_keys_from_markdown("") == []

    def test_extract_keys_from_markdown_no_headers(self):
        """Markdown without ## headers should return empty list."""
        assert RoutingMemoryAdapter._extract_keys_from_markdown("Just plain text\nNo headers") == []


# ============================================================================
# Import test
# ============================================================================


def test_routing_adapter_import():
    """Test that RoutingMemoryAdapter can be imported from the package."""
    from gepa.adapters.memory_adapter import RoutingMemoryAdapter

    assert RoutingMemoryAdapter is not None
    assert issubclass(RoutingMemoryAdapter, MemoryAdapter)
