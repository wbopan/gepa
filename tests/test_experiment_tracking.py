import shutil
import tempfile

import pytest

from gepa.logging.experiment_tracker import ExperimentTracker, create_experiment_tracker


def has_weave():
    """Check if weave and wandb are available."""
    try:
        import wandb  # noqa: F401
        import weave  # noqa: F401

        return True
    except ImportError:
        return False


class TestCreateExperimentTracker:
    """Test cases for create_experiment_tracker function."""

    def test_create_no_weave(self):
        """Test creating tracker with weave disabled."""
        tracker = create_experiment_tracker(
            use_weave=False,
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_weave is False

    def test_create_with_weave(self):
        """Test creating tracker with weave enabled."""
        tracker = create_experiment_tracker(
            use_weave=True,
            weave_project_name="test-project",
        )

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_weave is True
        assert tracker.weave_project_name == "test-project"

    def test_create_default_project_name(self):
        """Test that default project name is set when not provided."""
        tracker = create_experiment_tracker(use_weave=True)

        assert tracker.weave_project_name == "gepa-boost"

    def test_create_experiment_tracker_factory(self):
        """Test the create_experiment_tracker factory function."""
        # Test with weave disabled
        tracker1 = create_experiment_tracker(use_weave=False)
        assert isinstance(tracker1, ExperimentTracker)
        assert tracker1.use_weave is False

        # Test with weave enabled
        tracker2 = create_experiment_tracker(
            use_weave=True,
            weave_project_name="custom-project",
        )
        assert isinstance(tracker2, ExperimentTracker)
        assert tracker2.use_weave is True
        assert tracker2.weave_project_name == "custom-project"


class TestExperimentTrackerIntegration:
    """Integration tests for ExperimentTracker."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_weave_works(self):
        """Test that no weave configuration works."""
        tracker = create_experiment_tracker(use_weave=False)

        assert isinstance(tracker, ExperimentTracker)
        assert tracker.use_weave is False

        # Should work with context manager
        with tracker:
            tracker.log_metrics({"test": 1.0}, iteration=1)

        assert not tracker.is_active()

    def test_context_manager_no_weave(self):
        """Test context manager without weave."""
        tracker = ExperimentTracker(use_weave=False)

        # Test context manager workflow
        with tracker:
            tracker.log_metrics({"loss": 0.5}, iteration=1)
            tracker.log_metrics({"accuracy": 0.9}, iteration=2)

        # Should not be active after context exit
        assert not tracker.is_active()

    def test_context_manager_with_exception(self):
        """Test context manager with exception - should still clean up."""
        tracker = ExperimentTracker(use_weave=False)

        with pytest.raises(ValueError):
            with tracker:
                tracker.log_metrics({"test": 1.0}, iteration=1)
                raise ValueError("test exception")

        # Should not be active after exception
        assert not tracker.is_active()

    def test_metric_logging_variations(self):
        """Test various metric logging scenarios."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Test different metric types
            tracker.log_metrics({"loss": 0.5}, iteration=1)
            tracker.log_metrics({"accuracy": 0.9, "f1": 0.85}, iteration=2)
            tracker.log_metrics({"learning_rate": 0.001}, iteration=3)

            # Test without iteration
            tracker.log_metrics({"final_loss": 0.1})

            # Test with None iteration
            tracker.log_metrics({"test_metric": 42}, iteration=None)

        # No errors should occur

    def test_publish_prompt_no_weave(self):
        """Test publishing prompt with weave disabled (no-op)."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Should return None and not raise any errors when weave is disabled
            result = tracker.publish_prompt(
                content={"system": "You are a helpful assistant."},
                iteration=1,
                parent_ref=None,
                minibatch_score_before=0.5,
                minibatch_score_after=0.7,
                accepted=True,
                candidate_idx=0,
                valset_score=0.95,
            )
            assert result is None

    def test_publish_rejected_prompt_no_weave(self):
        """Test publishing rejected prompt with weave disabled (no-op)."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Should return None and not raise any errors when weave is disabled
            result = tracker.publish_prompt(
                content={"system": "You are a helpful assistant."},
                iteration=1,
                parent_ref=None,
                minibatch_score_before=0.5,
                minibatch_score_after=0.3,
                accepted=False,
            )
            assert result is None

    def test_get_prompt_ref_no_weave(self):
        """Test getting prompt ref with weave disabled."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Should return None when no refs have been published
            result = tracker.get_prompt_ref(0)
            assert result is None

    def test_log_final_results_no_weave(self):
        """Test logging final results with weave disabled (no-op)."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Should not raise any errors even though weave is disabled
            tracker.log_final_results(
                best_candidate={"system": "You are a helpful assistant."},
                best_candidate_idx=0,
                best_score=0.95,
                total_candidates=10,
                total_metric_calls=100,
            )

    def test_start_end_run_no_weave(self):
        """Test start_run and end_run with weave disabled."""
        tracker = ExperimentTracker(use_weave=False)

        # Should not raise any errors
        tracker.start_run()
        tracker.log_metrics({"test": 1.0}, iteration=1)
        tracker.end_run()

        assert not tracker.is_active()

    def test_is_active_no_weave(self):
        """Test is_active returns False when weave is disabled."""
        tracker = ExperimentTracker(use_weave=False)

        assert not tracker.is_active()

        with tracker:
            assert not tracker.is_active()

        assert not tracker.is_active()


@pytest.mark.skipif(not has_weave(), reason="weave not available")
class TestExperimentTrackerWithWeave:
    """Tests that require weave to be installed."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_weave_tracker_creation(self):
        """Test creating tracker with weave enabled."""
        tracker = ExperimentTracker(
            use_weave=True,
            weave_project_name="test-project",
        )

        assert tracker.use_weave is True
        assert tracker.weave_project_name == "test-project"
