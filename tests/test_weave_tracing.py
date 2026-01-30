# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for weave tracing utilities."""

from gepa.logging.weave_tracing import (
    add_call_feedback,
    configure_weave_tracing,
    is_weave_tracing_enabled,
    weave_op,
)


class TestConfigureWeaveTracing:
    """Tests for configure_weave_tracing function."""

    def test_enable_tracing(self):
        """Test enabling weave tracing."""
        configure_weave_tracing(enabled=True, client=None)
        assert is_weave_tracing_enabled() is True

        # Cleanup
        configure_weave_tracing(enabled=False, client=None)

    def test_disable_tracing(self):
        """Test disabling weave tracing."""
        configure_weave_tracing(enabled=False, client=None)
        assert is_weave_tracing_enabled() is False

    def test_toggle_tracing(self):
        """Test toggling weave tracing on and off."""
        configure_weave_tracing(enabled=True, client=None)
        assert is_weave_tracing_enabled() is True

        configure_weave_tracing(enabled=False, client=None)
        assert is_weave_tracing_enabled() is False


class TestWeaveOp:
    """Tests for weave_op decorator."""

    def setup_method(self):
        """Ensure tracing is disabled before each test."""
        configure_weave_tracing(enabled=False, client=None)

    def teardown_method(self):
        """Cleanup after each test."""
        configure_weave_tracing(enabled=False, client=None)

    def test_weave_op_disabled_passes_through(self):
        """Test that weave_op is a no-op when tracing is disabled."""
        call_count = 0

        @weave_op("test.operation")
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        result = test_function(1, 2)
        assert result == 3
        assert call_count == 1

    def test_weave_op_preserves_function_behavior(self):
        """Test that decorated function behaves correctly."""

        @weave_op("test.add")
        def add(a, b):
            return a + b

        @weave_op("test.multiply")
        def multiply(a, b):
            return a * b

        assert add(2, 3) == 5
        assert multiply(2, 3) == 6

    def test_weave_op_with_kwargs(self):
        """Test weave_op with keyword arguments."""

        @weave_op("test.greet")
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        assert greet("Alice") == "Hello, Alice!"
        assert greet("Bob", greeting="Hi") == "Hi, Bob!"

    def test_weave_op_without_name(self):
        """Test weave_op without explicit name."""

        @weave_op()
        def unnamed_function():
            return "result"

        assert unnamed_function() == "result"


class TestAddCallFeedback:
    """Tests for add_call_feedback function."""

    def setup_method(self):
        """Ensure tracing is disabled before each test."""
        configure_weave_tracing(enabled=False, client=None)

    def teardown_method(self):
        """Cleanup after each test."""
        configure_weave_tracing(enabled=False, client=None)

    def test_feedback_disabled_is_noop(self):
        """Test that add_call_feedback is a no-op when tracing is disabled."""
        # Should not raise any errors
        add_call_feedback(score=0.5)
        add_call_feedback(scores={"accuracy": 0.9, "f1": 0.85})
        add_call_feedback(score=0.5, scores={"test": 0.8})

    def test_feedback_with_none_values(self):
        """Test add_call_feedback with None values."""
        # Should not raise any errors
        add_call_feedback(score=None)
        add_call_feedback(scores=None)
        add_call_feedback()


class TestIntegration:
    """Integration tests combining multiple weave tracing features."""

    def setup_method(self):
        """Reset state before each test."""
        configure_weave_tracing(enabled=False, client=None)

    def teardown_method(self):
        """Cleanup after each test."""
        configure_weave_tracing(enabled=False, client=None)

    def test_multiple_decorated_calls(self):
        """Test multiple decorated function calls in sequence."""
        call_order = []

        @weave_op("test.step1")
        def step1():
            call_order.append("step1")
            return 1

        @weave_op("test.step2")
        def step2(x):
            call_order.append("step2")
            return x + 1

        @weave_op("test.step3")
        def step3(x):
            call_order.append("step3")
            return x * 2

        result = step3(step2(step1()))

        assert result == 4
        assert call_order == ["step1", "step2", "step3"]
