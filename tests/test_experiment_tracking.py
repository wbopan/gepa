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

        assert tracker.weave_project_name == "gepa-optimization"

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
            tracker.log_metrics({"test": 1.0}, step=1)

        assert not tracker.is_active()

    def test_context_manager_no_weave(self):
        """Test context manager without weave."""
        tracker = ExperimentTracker(use_weave=False)

        # Test context manager workflow
        with tracker:
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9}, step=2)

        # Should not be active after context exit
        assert not tracker.is_active()

    def test_context_manager_with_exception(self):
        """Test context manager with exception - should still clean up."""
        tracker = ExperimentTracker(use_weave=False)

        with pytest.raises(ValueError):
            with tracker:
                tracker.log_metrics({"test": 1.0}, step=1)
                raise ValueError("test exception")

        # Should not be active after exception
        assert not tracker.is_active()

    def test_metric_logging_variations(self):
        """Test various metric logging scenarios."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Test different metric types
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_metrics({"accuracy": 0.9, "f1": 0.85}, step=2)
            tracker.log_metrics({"learning_rate": 0.001}, step=3)

            # Test without step
            tracker.log_metrics({"final_loss": 0.1})

            # Test with None step
            tracker.log_metrics({"test_metric": 42}, step=None)

        # No errors should occur

    def test_log_prompt_artifact_no_weave(self):
        """Test logging prompt artifact with weave disabled (no-op)."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Should not raise any errors even though weave is disabled
            tracker.log_prompt_artifact(
                prompt={"system": "You are a helpful assistant."},
                candidate_idx=0,
                iteration=1,
                is_best=True,
                parent_idx=None,
                valset_score=0.95,
            )

    def test_log_score_distribution_no_weave(self):
        """Test logging score distribution with weave disabled (no-op)."""
        tracker = ExperimentTracker(use_weave=False)

        with tracker:
            # Should not raise any errors even though weave is disabled
            tracker.log_score_distribution(
                scores_by_val_id={0: 0.9, 1: 0.8, 2: 0.95},
                candidate_idx=0,
                iteration=1,
                objective_scores={"accuracy": 0.9},
            )

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
        tracker.log_metrics({"test": 1.0}, step=1)
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
