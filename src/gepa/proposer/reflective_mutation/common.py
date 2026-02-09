# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.core.callbacks import (
    CandidateSelectedEvent,
    EvaluationEndEvent,
    EvaluationSkippedEvent,
    EvaluationStartEvent,
    GEPACallback,
    MinibatchSampledEvent,
    ReflectiveDatasetBuiltEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector, ReflectionComponentSelector
from gepa.strategies.batch_sampler import BatchSampler, MetricLoggingBatchSampler

if TYPE_CHECKING:
    from gepa.logging.experiment_tracker import ExperimentTracker


@dataclass
class ProposalContext:
    """Intermediate state from proposal preparation, shared by different Proposers."""

    iteration: int

    # Parent information
    parent_idx: int
    parent_candidate: dict[str, str]
    parent_ids: list[int]
    is_seed_candidate: bool

    # Minibatch information
    subsample_ids: list[Any]
    minibatch: list[Any]

    # Parent evaluation result
    parent_eval: EvaluationBatch[Any, Any]

    # Reflection data
    components_to_update: list[str]
    reflective_dataset: dict[str, list[dict[str, Any]]]


def prepare_proposal_context(
    state: GEPAState,
    trainset: DataLoader[Any, Any],
    adapter: GEPAAdapter[Any, Any, Any],
    candidate_selector: CandidateSelector,
    module_selector: ReflectionComponentSelector,
    batch_sampler: BatchSampler[Any, Any],
    perfect_score: float,
    skip_perfect_score: bool,
    logger: Any,
    experiment_tracker: "ExperimentTracker",
    callbacks: list[GEPACallback] | None,
) -> ProposalContext | None:
    """
    Execute common preparation work before proposal.

    Extracted from ReflectiveMutationProposer.propose() to enable code reuse.

    Returns:
        ProposalContext if ready to propose, None if should skip this iteration.
    """
    i = state.i + 1

    # === 1. Select parent candidate ===
    curr_prog_id = candidate_selector.select_candidate_idx(state)
    curr_prog = state.program_candidates[curr_prog_id]
    state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
    logger.log(
        f"Iteration {i}: Selected program {curr_prog_id} score: {state.program_full_scores_val_set[curr_prog_id]}",
        header="select",
    )

    notify_callbacks(
        callbacks,
        "on_candidate_selected",
        CandidateSelectedEvent(
            iteration=i,
            candidate_idx=curr_prog_id,
            candidate=curr_prog,
            score=state.program_full_scores_val_set[curr_prog_id],
        ),
    )

    experiment_tracker.log_metrics({"candidate/selected_idx": curr_prog_id}, iteration=state.i)

    # === 2. Sample minibatch ===
    subsample_ids = batch_sampler.next_minibatch_ids(trainset, state)
    state.full_program_trace[-1]["subsample_ids"] = subsample_ids
    minibatch = trainset.fetch(subsample_ids)

    # Log metrics from weighted samplers if applicable
    if isinstance(batch_sampler, MetricLoggingBatchSampler):
        if batch_weights := batch_sampler.get_batch_weights():
            batch_metrics = {"train/batch_weight_avg": sum(batch_weights) / len(batch_weights)}
            logger.debug(f"Iteration {i}: Batch metrics: {batch_metrics}", header="metric")
            experiment_tracker.log_metrics(batch_metrics, iteration=state.i)
        if all_weights := batch_sampler.get_all_sample_weights():
            weights = list(all_weights.values())
            train_stats = {
                "train/weight_avg": sum(weights) / len(weights),
                "train/weight_max": max(weights),
                "train/weight_min": min(weights),
            }
            logger.debug(f"Iteration {i}: Train sample stats: {train_stats}", header="metric")
            experiment_tracker.log_metrics(train_stats, iteration=state.i)
            logger.debug(f"Iteration {i}: Logging {len(all_weights)} per-sample weights", header="metric")
            experiment_tracker.log_sample_weights_table(all_weights, iteration=state.i)

    notify_callbacks(
        callbacks,
        "on_minibatch_sampled",
        MinibatchSampledEvent(
            iteration=i,
            minibatch_ids=subsample_ids,
            trainset_size=len(trainset),
        ),
    )

    # === 3. Evaluate parent with traces ===
    curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
    is_seed_candidate = curr_prog_id == 0
    notify_callbacks(
        callbacks,
        "on_evaluation_start",
        EvaluationStartEvent(
            iteration=i,
            candidate_idx=curr_prog_id,
            batch_size=len(minibatch),
            capture_traces=True,
            parent_ids=curr_parent_ids,
            inputs=minibatch,
            is_seed_candidate=is_seed_candidate,
        ),
    )
    eval_curr = adapter.evaluate(minibatch, curr_prog, capture_traces=True)
    state.increment_evals(len(subsample_ids))
    state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
    notify_callbacks(
        callbacks,
        "on_evaluation_end",
        EvaluationEndEvent(
            iteration=i,
            candidate_idx=curr_prog_id,
            scores=eval_curr.scores,
            has_trajectories=bool(eval_curr.trajectories),
            parent_ids=curr_parent_ids,
            outputs=eval_curr.outputs,
            trajectories=eval_curr.trajectories,
            objective_scores=eval_curr.objective_scores,
            is_seed_candidate=is_seed_candidate,
        ),
    )

    # Update cache with current program evaluation results
    if state.evaluation_cache is not None:
        objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
        state.evaluation_cache.put_batch(
            curr_prog, subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list
        )

    # === 4. Skip checks ===
    if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
        logger.log(f"Iteration {i}: No trajectories captured. Skipping.", header="skip")
        notify_callbacks(
            callbacks,
            "on_evaluation_skipped",
            EvaluationSkippedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                reason="no_trajectories",
                scores=eval_curr.scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        return None

    if skip_perfect_score and all(s >= perfect_score for s in eval_curr.scores):
        logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.", header="skip")
        notify_callbacks(
            callbacks,
            "on_evaluation_skipped",
            EvaluationSkippedEvent(
                iteration=i,
                candidate_idx=curr_prog_id,
                reason="all_scores_perfect",
                scores=eval_curr.scores,
                is_seed_candidate=is_seed_candidate,
            ),
        )
        return None

    experiment_tracker.log_metrics({"train/batch_score_before": sum(eval_curr.scores)}, iteration=state.i)

    # === 5. Select components to update ===
    predictor_names_to_update = module_selector(
        state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
    )

    # === 6. Build reflective dataset ===
    reflective_dataset = adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)

    # Convert to concrete types for callback and storage
    reflective_dataset_concrete: dict[str, list[dict[str, Any]]] = {
        k: [dict(item) for item in v] for k, v in reflective_dataset.items()
    }

    notify_callbacks(
        callbacks,
        "on_reflective_dataset_built",
        ReflectiveDatasetBuiltEvent(
            iteration=i,
            candidate_idx=curr_prog_id,
            components=predictor_names_to_update,
            dataset=reflective_dataset_concrete,
        ),
    )

    return ProposalContext(
        iteration=i,
        parent_idx=curr_prog_id,
        parent_candidate=curr_prog,
        parent_ids=curr_parent_ids,
        is_seed_candidate=is_seed_candidate,
        subsample_ids=subsample_ids,
        minibatch=minibatch,
        parent_eval=eval_curr,
        components_to_update=predictor_names_to_update,
        reflective_dataset=reflective_dataset_concrete,
    )
