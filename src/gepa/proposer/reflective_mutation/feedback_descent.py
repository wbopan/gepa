# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass
class FailedAttempt:
    """Record of a failed proposal attempt."""

    attempt_number: int
    proposed_instruction: str
    score_before: float
    score_after: float
    failure_analysis: str
    improvement_direction: str


@dataclass
class FeedbackDescentConfig:
    """Configuration for Feedback Descent proposer."""

    max_attempts: int = 3
    improvement_threshold: float = 0.0


class PairwiseFeedbackGenerator:
    """Generates pairwise comparison feedback for failed attempts."""

    TEMPLATE = """Analyze why the modified prompt did not improve performance.

## Original Prompt
```
{original_prompt}
```
{original_truncation_note}
## Modified Prompt
```
{modified_prompt}
```
{modified_truncation_note}
## Score Change: {parent_score:.3f} -> {child_score:.3f} (delta: {delta:+.3f})

## Per-Example Comparison
{per_example_analysis}

Based on this analysis, provide:

<failure_analysis>
Explain why this modification failed to improve performance. Be specific about what went wrong.
</failure_analysis>

<improvement_direction>
Suggest specific changes to try next. Focus on concrete, actionable modifications.
</improvement_direction>
"""

    def __init__(self, lm: LanguageModel):
        self.lm = lm

    MAX_PROMPT_LENGTH = 20000
    MAX_OUTPUT_LENGTH = 5000  # Show substantial context
    TRUNCATE_HEAD = 2500  # Show first N chars (reasoning)
    TRUNCATE_TAIL = 2000  # Show last N chars (to capture final answer)

    @staticmethod
    def _escape_braces(text: str) -> str:
        """Escape curly braces to prevent str.format() conflicts."""
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _extract_output_content(output: Any) -> str:
        """Extract the actual content from an output object.

        Handles common output formats like dicts with 'full_assistant_response',
        'response', 'output', 'text', or 'content' keys.
        """
        if output is None:
            return "(no output)"

        # If it's a dict, try to extract the relevant field
        if isinstance(output, dict):
            # Common field names for LLM responses (NOT "answer" - that's for input data)
            for key in ["full_assistant_response", "response", "output", "text", "content"]:
                if key in output:
                    value = output[key]
                    if isinstance(value, str):
                        return value
                    # Recursively extract if nested
                    return PairwiseFeedbackGenerator._extract_output_content(value)
            # If no known key found, just return the dict as string but formatted better
            # Try to show a summary of keys
            keys = list(output.keys())
            if len(keys) == 1:
                return PairwiseFeedbackGenerator._extract_output_content(output[keys[0]])
            return str(output)

        # If it's a string, return as is
        if isinstance(output, str):
            return output

        # Otherwise convert to string
        return str(output)

    @staticmethod
    def _extract_input_content(inp: Any) -> str:
        """Extract the input/question content from a minibatch item.

        Handles common input formats like dicts with 'input', 'question', 'prompt' keys.
        This is different from _extract_output_content because input data has different key names.
        """
        if inp is None:
            return "(no input)"

        # If it's a dict, try to extract the input field
        if isinstance(inp, dict):
            # Common field names for input/question data (prioritize "input" over "answer")
            for key in ["input", "question", "prompt", "text", "content"]:
                if key in inp:
                    value = inp[key]
                    if isinstance(value, str):
                        return value
                    return PairwiseFeedbackGenerator._extract_input_content(value)
            # If no known key found, return dict as string (excluding 'answer' to avoid confusion)
            # Filter out answer-like keys for display
            display_dict = {k: v for k, v in inp.items() if k not in ["answer", "label", "target"]}
            if display_dict:
                return str(display_dict)
            return str(inp)

        # If it's a string, return as is
        if isinstance(inp, str):
            return inp

        # Otherwise convert to string
        return str(inp)

    def _smart_truncate(self, text: str, max_length: int = 0) -> str:
        """Truncate text showing both head and tail to preserve final answer.

        For LLM outputs, the final answer is often at the end, so we want to
        show both the beginning (reasoning) and end (conclusion).
        """
        if max_length <= 0:
            max_length = self.MAX_OUTPUT_LENGTH

        if len(text) <= max_length:
            return text

        # Show head + ... + tail
        head = text[: self.TRUNCATE_HEAD]
        tail = text[-self.TRUNCATE_TAIL :]
        return f"{head}\n\n[... truncated {len(text) - self.TRUNCATE_HEAD - self.TRUNCATE_TAIL} chars ...]\n\n{tail}"

    def generate(
        self,
        parent_instruction: str,
        child_instruction: str,
        parent_outputs: list[Any],
        child_outputs: list[Any],
        parent_scores: list[float],
        child_scores: list[float],
        minibatch: list[Any],
    ) -> tuple[str, str]:
        """Generate failure analysis and improvement direction.

        Returns:
            Tuple of (failure_analysis, improvement_direction)
        """
        per_example_parts = []
        for i, (p_out, c_out, p_s, c_s, inp) in enumerate(
            zip(parent_outputs, child_outputs, parent_scores, child_scores, minibatch, strict=False)
        ):
            delta = c_s - p_s
            status = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"

            # Format input (extract content, truncate if too long, escape braces)
            inp_content = self._extract_input_content(inp)
            inp_str = self._smart_truncate(inp_content, max_length=800)
            inp_str = self._escape_braces(inp_str)

            # Format outputs (extract content, smart truncate to show head+tail, escape braces)
            p_out_content = self._extract_output_content(p_out)
            c_out_content = self._extract_output_content(c_out)
            p_out_str = self._escape_braces(self._smart_truncate(p_out_content))
            c_out_str = self._escape_braces(self._smart_truncate(c_out_content))

            per_example_parts.append(
                f"""### Example {i + 1} [{status}]: {p_s:.3f} -> {c_s:.3f} (delta: {delta:+.3f})
**Input:** {inp_str}
**Original output:** {p_out_str}
**Modified output:** {c_out_str}
"""
            )

        parent_sum = sum(parent_scores)
        child_sum = sum(child_scores)

        # Handle truncation with notes
        original_truncated = len(parent_instruction) > self.MAX_PROMPT_LENGTH
        modified_truncated = len(child_instruction) > self.MAX_PROMPT_LENGTH

        # Escape braces in instructions to prevent str.format() conflicts
        # (instructions may contain template placeholders like {input}, {text}, etc.)
        original_prompt_safe = self._escape_braces(parent_instruction[: self.MAX_PROMPT_LENGTH])
        modified_prompt_safe = self._escape_braces(child_instruction[: self.MAX_PROMPT_LENGTH])

        prompt = self.TEMPLATE.format(
            original_prompt=original_prompt_safe,
            modified_prompt=modified_prompt_safe,
            original_truncation_note="*(truncated)*\n" if original_truncated else "",
            modified_truncation_note="*(truncated)*\n" if modified_truncated else "",
            parent_score=parent_sum,
            child_score=child_sum,
            delta=child_sum - parent_sum,
            per_example_analysis="\n".join(per_example_parts),
        )

        response = self.lm(prompt)

        failure = self._extract(response, "failure_analysis")
        direction = self._extract(response, "improvement_direction")
        return failure, direction

    @staticmethod
    def _extract(text: str, tag: str) -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""


class FeedbackDescentProposer(ProposeNewCandidate[DataId]):
    """
    Feedback Descent style proposer with iterative optimization.

    This proposer iteratively proposes modifications on a mini-batch,
    accumulating failure feedback until improvement is achieved or
    max_attempts is reached.
    """

    HISTORY_TEMPLATE = """
## Previous Failed Attempts

The following modifications were tried but did not improve performance:

{failed_attempts_text}

Based on these failures, try a different approach that addresses the identified issues.
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
        reflection_lm: LanguageModel,
        config: FeedbackDescentConfig,
        feedback_lm: LanguageModel | None = None,
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
        self.config = config
        self.callbacks = callbacks

        InstructionProposalSignature.validate_prompt_template(reflection_prompt_template)
        self.reflection_prompt_template = reflection_prompt_template

        self.feedback_generator = PairwiseFeedbackGenerator(feedback_lm or reflection_lm)

    MAX_INSTRUCTION_PREVIEW = 500

    def _format_failed_attempts(self, failed_attempts: list[FailedAttempt]) -> str:
        """Format failed attempts as text for inclusion in prompt."""
        parts = []
        for fa in failed_attempts:
            # Only add ellipsis if actually truncated
            instruction_preview = fa.proposed_instruction
            if len(instruction_preview) > self.MAX_INSTRUCTION_PREVIEW:
                instruction_preview = instruction_preview[: self.MAX_INSTRUCTION_PREVIEW] + "..."

            parts.append(f"""### Attempt {fa.attempt_number}
**Proposed modification:**
```
{instruction_preview}
```

**Result:** Score changed from {fa.score_before:.3f} to {fa.score_after:.3f}

**Why it failed:**
{fa.failure_analysis}

**Suggested direction:**
{fa.improvement_direction}
""")
        return "\n".join(parts)

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        failed_attempts: list[FailedAttempt],
    ) -> dict[str, str]:
        """Propose new texts with accumulated failure history."""
        if self.adapter.propose_new_texts is not None:
            # Adapter provides its own proposal function
            return self.adapter.propose_new_texts(candidate, reflective_dataset, components_to_update)

        new_texts: dict[str, str] = {}
        for name in components_to_update:
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(f"Component '{name}' is not in reflective dataset. Skipping.", header="skip")
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]

            # Build prompt with failed attempts history if any
            prompt_template = self.reflection_prompt_template
            if failed_attempts:
                history_text = self._format_failed_attempts(failed_attempts)
                history_section = self.HISTORY_TEMPLATE.format(failed_attempts_text=history_text)

                if prompt_template is None:
                    # Use default template and append history
                    prompt_template = InstructionProposalSignature.default_prompt_template + "\n\n" + history_section
                else:
                    # Append history to custom template
                    prompt_template = prompt_template + "\n\n" + history_section

            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": prompt_template,
                },
            )["new_instruction"]

        return new_texts

    @weave.op(name="gepa.propose.feedback_descent")
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

        # Feedback Descent iteration loop
        failed_attempts: list[FailedAttempt] = []
        parent_scores = ctx.parent_eval.scores
        parent_sum = sum(parent_scores)

        best_candidate = ctx.parent_candidate
        best_scores = parent_scores
        best_outputs = ctx.parent_eval.outputs

        for attempt in range(1, self.config.max_attempts + 1):
            self.logger.log(
                f"Iteration {ctx.iteration}: Feedback Descent attempt {attempt}/{self.config.max_attempts}",
                header="propose",
            )

            # Generate proposal with history
            try:
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

                new_texts = self.propose_new_texts(
                    ctx.parent_candidate,
                    ctx.reflective_dataset,
                    ctx.components_to_update,
                    failed_attempts,
                )

                notify_callbacks(
                    self.callbacks,
                    "on_proposal_end",
                    ProposalEndEvent(
                        iteration=ctx.iteration,
                        new_instructions=new_texts,
                    ),
                )

                if not new_texts:
                    self.logger.log(f"Attempt {attempt}: No texts proposed, skipping", header="skip")
                    continue

                for pname, text in new_texts.items():
                    self.logger.log(f"Attempt {attempt}: Proposed new text for {pname}", header="propose")
                    if hasattr(self.logger, "show"):
                        self.logger.show(text, title=f"Proposed: {pname}")

            except Exception as e:
                self.logger.log(f"Attempt {attempt}: Exception during proposal: {e}", header="error")
                import traceback

                self.logger.log(traceback.format_exc(), header="error")
                continue

            # Create new candidate
            new_candidate = ctx.parent_candidate.copy()
            for pname, text in new_texts.items():
                if pname not in new_candidate:
                    self.logger.log(f"Component {pname} not in candidate, skipping", header="error")
                    continue
                new_candidate[pname] = text

            # Evaluate new candidate
            def evaluator(b, c):
                r = self.adapter.evaluate(b, c, capture_traces=False)
                return r.outputs, r.scores, list(r.objective_scores) if r.objective_scores else None

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
            new_outputs = [outputs_by_id[eid] for eid in ctx.subsample_ids]
            state.increment_evals(actual_evals_count)

            notify_callbacks(
                self.callbacks,
                "on_evaluation_end",
                EvaluationEndEvent(
                    iteration=ctx.iteration,
                    candidate_idx=None,
                    scores=new_scores,
                    has_trajectories=False,
                    parent_ids=[ctx.parent_idx],
                    outputs=new_outputs,
                    trajectories=None,
                    objective_scores=[objective_by_id[eid] for eid in ctx.subsample_ids] if objective_by_id else None,
                    is_seed_candidate=False,
                ),
            )

            new_sum = sum(new_scores)
            improvement = new_sum - parent_sum

            # Check if improved
            if improvement > self.config.improvement_threshold:
                self.logger.log(
                    f"Feedback Descent: Improved on attempt {attempt}! "
                    f"{parent_sum:.3f} -> {new_sum:.3f} (+{improvement:.3f})",
                    header="accept",
                )
                best_candidate = new_candidate
                best_scores = new_scores
                best_outputs = new_outputs
                break

            # No improvement -> generate pairwise feedback
            self.logger.log(
                f"Attempt {attempt}: No improvement ({parent_sum:.3f} -> {new_sum:.3f})",
                header="reject",
            )

            # Generate feedback for the first component that was updated
            component = ctx.components_to_update[0] if ctx.components_to_update else None
            if component and component in new_texts:
                try:
                    failure, direction = self.feedback_generator.generate(
                        parent_instruction=ctx.parent_candidate[component],
                        child_instruction=new_texts[component],
                        parent_outputs=ctx.parent_eval.outputs,
                        child_outputs=new_outputs,
                        parent_scores=parent_scores,
                        child_scores=new_scores,
                        minibatch=ctx.minibatch,
                    )
                    failed_attempts.append(
                        FailedAttempt(
                            attempt_number=attempt,
                            proposed_instruction=new_texts[component],
                            score_before=parent_sum,
                            score_after=new_sum,
                            failure_analysis=failure,
                            improvement_direction=direction,
                        )
                    )
                except Exception as e:
                    self.logger.log(f"Failed to generate feedback: {e}", header="error")

            # Track best even if no improvement over parent (strict > to prefer earlier attempts on tie)
            if new_sum > sum(best_scores):
                best_candidate = new_candidate
                best_scores = new_scores
                best_outputs = new_outputs

        # Log metrics
        improved = sum(best_scores) > parent_sum
        self.experiment_tracker.log_metrics(
            {
                "feedback_descent/attempts": len(failed_attempts) + (1 if improved else 0),
                "feedback_descent/improved": int(improved),
                "train/batch_score_after": sum(best_scores),
            },
            iteration=state.i,
        )

        state.full_program_trace[-1]["new_subsample_scores"] = best_scores

        # Add weave feedback
        add_call_feedback(
            scores={
                "mini_batch/before": parent_sum,
                "mini_batch/after": sum(best_scores),
            },
        )

        return CandidateProposal(
            candidate=best_candidate,
            parent_program_ids=[ctx.parent_idx],
            subsample_indices=ctx.subsample_ids,
            subsample_scores_before=parent_scores,
            subsample_scores_after=best_scores,
            tag="feedback_descent",
            metadata={
                "iteration_id": ctx.iteration,
                "attempts": len(failed_attempts) + (1 if improved else 0),
                "improved": improved,
                "failed_attempts": [
                    {
                        "attempt": fa.attempt_number,
                        "score_before": fa.score_before,
                        "score_after": fa.score_after,
                    }
                    for fa in failed_attempts
                ],
                "subsample_score_before": parent_sum,
                "subsample_score_after": sum(best_scores),
                "subsample_inputs": list(ctx.minibatch),
                "outputs_before": ctx.parent_eval.outputs,
                "outputs_after": best_outputs,
            },
        )
