# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Tests for the Memory Adapter.

Tests cover:
- memory_store: XML parsing, serialization, validation, editing, markdown formatting
- memory_adapter: evaluation context, edit parsing, proposal, reflective dataset, evaluate
"""

import json

import pytest

from gepa.adapters.memory_adapter.memory_store import (
    EditOperation,
    MemoryEntry,
    apply_edit,
    format_memory_as_markdown,
    parse_memory_xml,
    serialize_memory,
    validate_memory_xml,
)
from gepa.core.adapter import EvaluationBatch

# ============================================================================
# memory_store tests
# ============================================================================


class TestParseMemoryXML:
    """Tests for parse_memory_xml."""

    def test_single_entry(self):
        xml = '<memory>\n<entry key="greeting">Hello world</entry>\n</memory>'
        entries = parse_memory_xml(xml)
        assert len(entries) == 1
        assert entries[0].key == "greeting"
        assert entries[0].content == "Hello world"

    def test_multiple_entries(self):
        xml = '<memory>\n<entry key="a">Alpha</entry>\n<entry key="b">Beta</entry>\n</memory>'
        entries = parse_memory_xml(xml)
        assert len(entries) == 2
        assert entries[0].key == "a"
        assert entries[1].key == "b"

    def test_empty_memory(self):
        xml = "<memory>\n</memory>"
        entries = parse_memory_xml(xml)
        assert len(entries) == 0

    def test_multiline_content(self):
        xml = '<memory>\n<entry key="notes">Line 1\nLine 2\nLine 3</entry>\n</memory>'
        entries = parse_memory_xml(xml)
        assert len(entries) == 1
        assert "Line 1\nLine 2\nLine 3" == entries[0].content

    def test_content_containing_close_tag(self):
        xml = '<memory>\n<entry key="xml_example">Use </entry> to close tags</entry>\n</memory>'
        entries = parse_memory_xml(xml)
        assert len(entries) == 1
        assert entries[0].key == "xml_example"
        assert entries[0].content == "Use </entry> to close tags"

    def test_malformed_raises(self):
        with pytest.raises(ValueError, match="<memory>"):
            parse_memory_xml("not xml at all")

    def test_missing_wrapper_raises(self):
        with pytest.raises(ValueError, match="<memory>"):
            parse_memory_xml('<entry key="a">value</entry>')


class TestSerializeMemory:
    """Tests for serialize_memory."""

    def test_roundtrip(self):
        entries = [
            MemoryEntry(key="a", content="Alpha"),
            MemoryEntry(key="b", content="Beta"),
        ]
        xml = serialize_memory(entries)
        parsed = parse_memory_xml(xml)
        assert len(parsed) == 2
        assert parsed[0].key == "a"
        assert parsed[0].content == "Alpha"
        assert parsed[1].key == "b"
        assert parsed[1].content == "Beta"

    def test_empty_entries(self):
        xml = serialize_memory([])
        assert xml == "<memory>\n</memory>"
        entries = parse_memory_xml(xml)
        assert len(entries) == 0

    def test_single_entry(self):
        xml = serialize_memory([MemoryEntry(key="x", content="value")])
        assert '<entry key="x">value</entry>' in xml
        assert xml.startswith("<memory>")
        assert xml.endswith("</memory>")


class TestApplyEdit:
    """Tests for apply_edit."""

    def test_update_content(self):
        xml = '<memory>\n<entry key="fact">The sky is green</entry>\n</memory>'
        edit = EditOperation(old_string="The sky is green", new_string="The sky is blue")
        result = apply_edit(xml, edit)
        entries = parse_memory_xml(result)
        assert entries[0].content == "The sky is blue"

    def test_create_entry(self):
        xml = "<memory>\n</memory>"
        edit = EditOperation(
            old_string="</memory>",
            new_string='<entry key="new_fact">Important info</entry>\n</memory>',
        )
        result = apply_edit(xml, edit)
        entries = parse_memory_xml(result)
        assert len(entries) == 1
        assert entries[0].key == "new_fact"

    def test_delete_entry(self):
        xml = '<memory>\n<entry key="old">remove me</entry>\n<entry key="keep">stay</entry>\n</memory>'
        edit = EditOperation(
            old_string='<entry key="old">remove me</entry>\n',
            new_string="",
        )
        result = apply_edit(xml, edit)
        entries = parse_memory_xml(result)
        assert len(entries) == 1
        assert entries[0].key == "keep"

    def test_rename_key(self):
        xml = '<memory>\n<entry key="old_name">content</entry>\n</memory>'
        edit = EditOperation(
            old_string='key="old_name"',
            new_string='key="new_name"',
        )
        result = apply_edit(xml, edit)
        entries = parse_memory_xml(result)
        assert entries[0].key == "new_name"

    def test_not_found_raises(self):
        xml = "<memory>\n</memory>"
        edit = EditOperation(old_string="nonexistent text", new_string="replacement")
        with pytest.raises(ValueError, match="not found"):
            apply_edit(xml, edit)

    def test_ambiguous_raises(self):
        xml = '<memory>\n<entry key="a">same</entry>\n<entry key="b">same</entry>\n</memory>'
        edit = EditOperation(old_string="same", new_string="different")
        with pytest.raises(ValueError, match="ambiguous"):
            apply_edit(xml, edit)

    def test_invalid_result_raises(self):
        xml = '<memory>\n<entry key="a">content</entry>\n</memory>'
        # This edit breaks the XML structure by removing the closing tag
        edit = EditOperation(old_string="</memory>", new_string="")
        with pytest.raises(ValueError):
            apply_edit(xml, edit)


class TestValidateMemoryXML:
    """Tests for validate_memory_xml."""

    def test_valid(self):
        xml = '<memory>\n<entry key="a">Alpha</entry>\n</memory>'
        assert validate_memory_xml(xml) is True

    def test_duplicate_keys_raises(self):
        xml = '<memory>\n<entry key="a">1</entry>\n<entry key="a">2</entry>\n</memory>'
        with pytest.raises(ValueError, match="Duplicate"):
            validate_memory_xml(xml)

    def test_missing_wrapper_raises(self):
        with pytest.raises(ValueError, match="<memory>"):
            validate_memory_xml("no wrapper here")

    def test_empty_valid(self):
        assert validate_memory_xml("<memory>\n</memory>") is True


class TestFormatMemoryAsMarkdown:
    """Tests for format_memory_as_markdown."""

    def test_with_entries(self):
        entries = [
            MemoryEntry(key="greeting", content="Hello"),
            MemoryEntry(key="farewell", content="Goodbye"),
        ]
        md = format_memory_as_markdown(entries)
        assert "## greeting" in md
        assert "Hello" in md
        assert "## farewell" in md
        assert "Goodbye" in md

    def test_empty_entries(self):
        md = format_memory_as_markdown([])
        assert md == ""


# ============================================================================
# memory_adapter tests
# ============================================================================

from gepa.adapters.memory_adapter.memory_adapter import (  # noqa: E402
    MemoryAdapter,
    MemoryDataInst,
    MemoryOutput,
    MemoryTrajectory,
)

# ---- Fixtures ----


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
def mock_reflection_model():
    """Mock reflection model callable."""

    def model(messages):
        return json.dumps(
            {"old_string": "</memory>", "new_string": '<entry key="fact">42 is the answer</entry>\n</memory>'}
        )

    return model


@pytest.fixture
def adapter(simple_evaluator, mock_task_model, mock_reflection_model):
    """Create a MemoryAdapter with mock models."""
    return MemoryAdapter(
        task_model=mock_task_model,
        reflection_model=mock_reflection_model,
        evaluator=simple_evaluator,
        base_system_prompt="You are a helpful assistant.",
        max_entries=50,
        max_retries=2,
    )


@pytest.fixture
def sample_batch() -> list[MemoryDataInst]:
    """Sample batch of data."""
    return [
        {"input": "What is the meaning of life?", "answer": "42"},
        {"input": "What color is the sky?", "answer": "blue"},
    ]


@pytest.fixture
def seed_candidate() -> dict[str, str]:
    """Sample seed candidate with empty memory."""
    return {"memory": "<memory>\n</memory>"}


@pytest.fixture
def populated_candidate() -> dict[str, str]:
    """Sample candidate with memory entries."""
    return {
        "memory": '<memory>\n<entry key="fact1">The meaning of life is 42</entry>\n<entry key="fact2">The sky is blue</entry>\n</memory>'
    }


# ---- Tests ----


class TestBuildEvaluationContext:
    """Tests for _build_evaluation_context."""

    def test_with_entries(self, adapter):
        xml = '<memory>\n<entry key="fact">Important info</entry>\n</memory>'
        prompt, md = adapter._build_evaluation_context(xml)
        assert "You are a helpful assistant." in prompt
        assert "Knowledge Memory" in prompt
        assert "## fact" in md
        assert "Important info" in md

    def test_empty_memory(self, adapter):
        xml = "<memory>\n</memory>"
        prompt, md = adapter._build_evaluation_context(xml)
        assert prompt == "You are a helpful assistant."
        assert md == ""

    def test_invalid_xml_fallback(self, adapter):
        prompt, md = adapter._build_evaluation_context("not valid xml")
        assert prompt == "You are a helpful assistant."
        assert md == ""


class TestParseEditResponse:
    """Tests for _parse_edit_response."""

    def test_raw_json(self):
        output = '{"old_string": "hello", "new_string": "world"}'
        edit = MemoryAdapter._parse_edit_response(output)
        assert edit.old_string == "hello"
        assert edit.new_string == "world"

    def test_code_fenced_json(self):
        output = 'Here is my edit:\n```json\n{"old_string": "hello", "new_string": "world"}\n```'
        edit = MemoryAdapter._parse_edit_response(output)
        assert edit.old_string == "hello"
        assert edit.new_string == "world"

    def test_code_fenced_no_language(self):
        output = '```\n{"old_string": "a", "new_string": "b"}\n```'
        edit = MemoryAdapter._parse_edit_response(output)
        assert edit.old_string == "a"
        assert edit.new_string == "b"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            MemoryAdapter._parse_edit_response("not json at all")

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing"):
            MemoryAdapter._parse_edit_response('{"old_string": "hello"}')

    def test_missing_new_string_raises(self):
        with pytest.raises(ValueError, match="missing"):
            MemoryAdapter._parse_edit_response('{"new_string": "hello"}')


class TestProposeNewTexts:
    """Tests for propose_new_texts."""

    def test_successful_edit(self, adapter, seed_candidate):
        reflective_dataset = {
            "memory": [
                {
                    "Inputs": {"user_input": "What is 42?", "expected_answer": "The answer"},
                    "Generated Outputs": "I don't know",
                    "Feedback": "Should mention 42",
                }
            ]
        }
        result = adapter._propose_new_texts(seed_candidate, reflective_dataset, ["memory"])
        assert "memory" in result
        entries = parse_memory_xml(result["memory"])
        assert len(entries) == 1
        assert entries[0].key == "fact"

    def test_skips_non_memory_components(self, adapter, seed_candidate):
        result = adapter._propose_new_texts(seed_candidate, {}, ["other_component"])
        assert result == {}

    def test_skips_when_memory_not_in_components(self, adapter, seed_candidate):
        reflective_dataset = {"memory": [{"Inputs": {}, "Generated Outputs": "", "Feedback": ""}]}
        result = adapter._propose_new_texts(seed_candidate, reflective_dataset, ["other"])
        assert result == {}

    def test_retry_on_failure(self, simple_evaluator, mock_task_model):
        call_count = 0

        def failing_then_succeeding_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return "not json"
            return json.dumps(
                {
                    "old_string": "</memory>",
                    "new_string": '<entry key="x">y</entry>\n</memory>',
                }
            )

        adapter = MemoryAdapter(
            task_model=mock_task_model,
            reflection_model=failing_then_succeeding_model,
            evaluator=simple_evaluator,
            max_retries=2,
        )

        result = adapter._propose_new_texts(
            {"memory": "<memory>\n</memory>"},
            {"memory": [{"Inputs": {}, "Generated Outputs": "", "Feedback": ""}]},
            ["memory"],
        )
        assert "memory" in result
        assert call_count == 3

    def test_raises_after_max_retries(self, simple_evaluator, mock_task_model):
        def always_failing_model(messages):
            return "not valid json ever"

        adapter = MemoryAdapter(
            task_model=mock_task_model,
            reflection_model=always_failing_model,
            evaluator=simple_evaluator,
            max_retries=1,
        )

        with pytest.raises(RuntimeError, match="All 2 edit proposal attempts failed"):
            adapter._propose_new_texts(
                {"memory": "<memory>\n</memory>"},
                {"memory": [{"Inputs": {}, "Generated Outputs": "", "Feedback": ""}]},
                ["memory"],
            )

    def test_max_entries_exceeded(self, simple_evaluator, mock_task_model):
        def model_adds_entry(messages):
            return json.dumps(
                {
                    "old_string": "</memory>",
                    "new_string": '<entry key="new">content</entry>\n</memory>',
                }
            )

        adapter = MemoryAdapter(
            task_model=mock_task_model,
            reflection_model=model_adds_entry,
            evaluator=simple_evaluator,
            max_entries=1,
            max_retries=0,
        )

        # Already have 1 entry, trying to add another should exceed max_entries=1
        candidate = {"memory": '<memory>\n<entry key="existing">data</entry>\n</memory>'}

        with pytest.raises(RuntimeError, match="edit proposal attempts failed"):
            adapter._propose_new_texts(
                candidate,
                {"memory": [{"Inputs": {}, "Generated Outputs": "", "Feedback": ""}]},
                ["memory"],
            )


class TestMakeReflectiveDataset:
    """Tests for make_reflective_dataset."""

    def test_correct_format(self, adapter, seed_candidate):
        trajectories: list[MemoryTrajectory] = [
            {
                "data": {"input": "What is 42?", "answer": "The answer to everything"},
                "system_prompt": "You are helpful.",
                "memory_markdown": "",
                "full_assistant_response": "I don't know",
                "score": 0.0,
                "feedback": "Should know about 42",
            }
        ]
        outputs: list[MemoryOutput] = [{"full_assistant_response": "I don't know"}]

        eval_batch: EvaluationBatch[MemoryTrajectory, MemoryOutput] = EvaluationBatch(
            outputs=outputs,
            scores=[0.0],
            trajectories=trajectories,
        )

        result = adapter.make_reflective_dataset(seed_candidate, eval_batch, ["memory"])
        assert "memory" in result
        assert len(result["memory"]) == 1

        example = result["memory"][0]
        assert "Inputs" in example
        assert "Generated Outputs" in example
        assert "Feedback" in example
        assert example["Inputs"]["user_input"] == "What is 42?"
        assert example["Inputs"]["expected_answer"] == "The answer to everything"
        assert example["Generated Outputs"] == "I don't know"
        assert example["Feedback"] == "Should know about 42"

    def test_ignores_non_memory(self, adapter, seed_candidate):
        eval_batch: EvaluationBatch[MemoryTrajectory, MemoryOutput] = EvaluationBatch(
            outputs=[],
            scores=[],
            trajectories=[],
        )

        result = adapter.make_reflective_dataset(seed_candidate, eval_batch, ["other_component"])
        assert "memory" not in result


class TestEvaluate:
    """Tests for evaluate."""

    def test_basic_evaluate(self, adapter, sample_batch, seed_candidate):
        result = adapter.evaluate(sample_batch, seed_candidate)
        assert len(result.outputs) == 2
        assert len(result.scores) == 2
        assert result.trajectories is None

    def test_evaluate_with_traces(self, adapter, sample_batch, seed_candidate):
        result = adapter.evaluate(sample_batch, seed_candidate, capture_traces=True)
        assert result.trajectories is not None
        assert len(result.trajectories) == 2
        for traj in result.trajectories:
            assert "data" in traj
            assert "system_prompt" in traj
            assert "full_assistant_response" in traj
            assert "score" in traj
            assert "feedback" in traj

    def test_evaluate_scores(self, adapter, seed_candidate):
        # The mock model returns "The answer is 42."
        # First item answer is "42" -> should match
        # Second item answer is "blue" -> should not match
        batch: list[MemoryDataInst] = [
            {"input": "What?", "answer": "42"},
            {"input": "Color?", "answer": "blue"},
        ]
        result = adapter.evaluate(batch, seed_candidate)
        assert result.scores[0] == 1.0  # "42" is in "The answer is 42."
        assert result.scores[1] == 0.0  # "blue" is not in "The answer is 42."

    def test_evaluate_failure_handling(self, seed_candidate):
        def failing_evaluator(data, response):
            raise RuntimeError("Evaluator crashed")

        def task_model(messages):
            return "response"

        adapter = MemoryAdapter(
            task_model=task_model,
            reflection_model=task_model,
            evaluator=failing_evaluator,
            failure_score=0.0,
        )

        batch: list[MemoryDataInst] = [{"input": "test", "answer": "test"}]
        result = adapter.evaluate(batch, seed_candidate)
        assert len(result.outputs) == 1
        assert result.scores[0] == 0.0


# ============================================================================
# Type import tests
# ============================================================================


def test_types_import():
    """Test that types can be imported from the package."""
    from gepa.adapters.memory_adapter import MemoryAdapter, MemoryDataInst, MemoryOutput, MemoryTrajectory

    assert MemoryAdapter is not None
    assert MemoryDataInst is not None
    assert MemoryOutput is not None
    assert MemoryTrajectory is not None
