# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from collections.abc import Mapping, Sequence
from typing import Any

import weave

from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.callbacks import (
    EvaluationEndEvent,
    EvaluationStartEvent,
    GEPACallback,
    ProposalEndEvent,
    ProposalStartEvent,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader, ensure_loader
from gepa.core.state import GEPAState
from gepa.logging.weave_tracing import add_call_feedback
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from gepa.proposer.reflective_mutation.common import prepare_proposal_context
from gepa.strategies.batch_sampler import BatchSampler
from gepa.strategies.instruction_proposal import InstructionProposalSignature


class ReflectiveMutationProposer(ProposeNewCandidate[DataId]):
    """
    Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
        self,
        logger: Any,
        trainset: list[DataInst] | DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float,
        skip_perfect_score: bool,
        experiment_tracker: Any,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | None = None,
        callbacks: list[GEPACallback] | None = None,
    ):
        self.logger = logger
        self.trainset = ensure_loader(trainset)
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.experiment_tracker = experiment_tracker
        self.reflection_lm = reflection_lm
        self.callbacks = callbacks

        InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)
        self.reflection_prompt_template = reflection_prompt_template

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        if self.reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")
        new_texts: dict[str, str] = {}
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.", header="skip")
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": self.reflection_prompt_template,
                },
            )["new_instruction"]
        return new_texts

    @weave.op(name="gepa.propose.reflective_mutation")
    def propose(self, state: GEPAState) -> CandidateProposal | None:
        # Use shared function for proposal preparation
        ctx = prepare_proposal_context(
            state=state,
            trainset=self.trainset,
            adapter=self.adapter,
            candidate_selector=self.candidate_selector,
            module_selector=self.module_selector,
            batch_sampler=self.batch_sampler,
            perfect_score=self.perfect_score,
            skip_perfect_score=self.skip_perfect_score,
            logger=self.logger,
            experiment_tracker=self.experiment_tracker,
            callbacks=self.callbacks,
        )
        if ctx is None:
            return None

        # Build reflective dataset and propose texts
        try:
            # Notify proposal start
            notify_callbacks(
                self.callbacks,
                "on_proposal_start",
                ProposalStartEvent(
                    iteration=ctx.iteration,
                    parent_candidate=ctx.parent_candidate,
                    components=ctx.components_to_update,
                    reflective_dataset=ctx.reflective_dataset,
                ),
            )

            new_texts = self.propose_new_texts(ctx.parent_candidate, ctx.reflective_dataset, ctx.components_to_update)

            # Notify proposal end
            notify_callbacks(
                self.callbacks,
                "on_proposal_end",
                ProposalEndEvent(
                    iteration=ctx.iteration,
                    new_instructions=new_texts,
                ),
            )

            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {ctx.iteration}: Proposed new text for {pname}", header="propose")
                if hasattr(self.logger, "show"):
                    self.logger.show(text, title=f"Proposed: {pname}")
                else:
                    self.logger.log(text)
        except Exception as e:
            self.logger.log(f"Iteration {ctx.iteration}: Exception during reflection/proposal: {e}", header="error")
            import traceback

            self.logger.log(traceback.format_exc(), header="error")
            return None

        # Create candidate, evaluate on same minibatch (no need to capture traces)
        new_candidate = ctx.parent_candidate.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        def evaluator(b, c):
            r = self.adapter.evaluate(b, c, capture_traces=False)
            return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

        # Evaluate new candidate (not yet in state)
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=ctx.iteration,
                candidate_idx=None,
                batch_size=len(ctx.minibatch),
                capture_traces=False,
                parent_ids=[ctx.parent_idx],
                inputs=ctx.minibatch,
                is_seed_candidate=False,
            ),
        )

        outputs_by_id, scores_by_id, objective_by_id, actual_evals_count = state.cached_evaluate_full(
            new_candidate, ctx.subsample_ids, self.trainset.fetch, evaluator
        )
        new_scores = [scores_by_id[eid] for eid in ctx.subsample_ids]
        outputs = [outputs_by_id[eid] for eid in ctx.subsample_ids]

        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=ctx.iteration,
                candidate_idx=None,
                scores=new_scores,
                has_trajectories=False,
                parent_ids=[ctx.parent_idx],
                outputs=outputs,
                trajectories=None,
                objective_scores=[objective_by_id[eid] for eid in ctx.subsample_ids] if objective_by_id else None,
                is_seed_candidate=False,
            ),
        )

        state.increment_evals(actual_evals_count)
        state.full_program_trace[-1]["new_subsample_scores"] = new_scores

        new_sum = sum(new_scores)
        self.experiment_tracker.log_metrics({"train/batch_score_after": new_sum}, iteration=state.i)

        # Add feedback with subsample scores
        add_call_feedback(
            scores={
                "mini_batch/before": sum(ctx.parent_eval.scores),
                "mini_batch/after": new_sum,
            },
        )

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[ctx.parent_idx],
            subsample_indices=ctx.subsample_ids,
            subsample_scores_before=ctx.parent_eval.scores,
            subsample_scores_after=new_scores,
            tag="reflective_mutation",
            metadata={
                "iteration_id": ctx.iteration,
                "subsample_score_before": sum(ctx.parent_eval.scores),
                "subsample_score_after": new_sum,
                "accepted": new_sum > sum(ctx.parent_eval.scores),
                "subsample_inputs": list(ctx.minibatch),
                "outputs_before": ctx.parent_eval.outputs,
                "outputs_after": [outputs_by_id[sid] for sid in ctx.subsample_ids],
            },
        )
