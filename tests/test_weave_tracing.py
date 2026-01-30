# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for weave tracing utilities."""

import weave

from gepa.logging.weave_tracing import add_call_feedback


class TestAddCallFeedback:
    """Tests for add_call_feedback function."""

    def test_feedback_outside_call_is_noop(self):
        """Test that add_call_feedback is a no-op when not in a weave call."""
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

    def test_multiple_decorated_calls(self):
        """Test multiple decorated function calls in sequence."""
        call_order = []

        @weave.op(name="test.step1")
        def step1():
            call_order.append("step1")
            return 1

        @weave.op(name="test.step2")
        def step2(x):
            call_order.append("step2")
            return x + 1

        @weave.op(name="test.step3")
        def step3(x):
            call_order.append("step3")
            return x * 2

        result = step3(step2(step1()))

        assert result == 4
        assert call_order == ["step1", "step2", "step3"]
