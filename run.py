#!/usr/bin/env python3
"""
GPQA Diamond experiment with GEPA using OpenRouter + DeepSeek V3.2.
Run from project root: uv run run.py
"""

import os
import sys
import warnings

import litellm
from litellm.integrations.custom_logger import CustomLogger

import gepa
from gepa import get_logger
from gepa.examples.gpqa import init_dataset

# Suppress noisy output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
litellm.suppress_debug_info = True

# Initialize the logger
logger = get_logger()


class LiteLLMCacheLogger(CustomLogger):
    """Logs LiteLLM request timing and cache hits/misses."""

    def log_pre_api_call(self, model, messages, kwargs):
        """Called before each API request is sent."""
        from datetime import datetime

        logger.debug(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {model}", header="request")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called when a request completes successfully."""
        cache_hit = kwargs.get("cache_hit", False)
        model = kwargs.get("model", "unknown")
        duration = (end_time - start_time).total_seconds()
        header = "cache hit" if cache_hit else "cache miss"
        logger.debug(f"[{end_time.strftime('%H:%M:%S.%f')[:-3]}] {model} ({duration:.2f}s)", header=header)


# Register litellm callback
litellm.callbacks = [LiteLLMCacheLogger()]


class VerboseCallback:
    """Prints detailed progress during optimization using RichLogger."""

    def __init__(self):
        self.logger = get_logger()
        self.indent_logger = self.logger.indent()

    def on_optimization_start(self, event):
        self.logger.log(
            f"Optimization started: {event['trainset_size']} train, {event['valset_size']} val examples",
            header="start",
        )

    def on_iteration_start(self, event):
        self.logger.log(f"Starting iteration {event['iteration']}...", header="iter")

    def on_candidate_selected(self, event):
        self.indent_logger.log(
            f"Candidate {event['candidate_idx']} selected (score: {event['score']:.2%})", header="select"
        )

    def on_minibatch_sampled(self, event):
        self.indent_logger.log(f"Minibatch of {len(event['minibatch_ids'])} examples sampled", header="sample")

    def on_evaluation_start(self, event):
        self.indent_logger.log(f"Evaluating batch of {event['batch_size']} examples...", header="eval")

    def on_evaluation_end(self, event):
        avg = sum(event["scores"]) / len(event["scores"]) if event["scores"] else 0
        self.indent_logger.log(f"Done. Avg score: {avg:.2%}", header="eval")

    def on_candidate_accepted(self, event):
        self.indent_logger.log(
            f"New candidate {event['new_candidate_idx']} accepted (score: {event['new_score']:.2f})", header="accept"
        )

    def on_candidate_rejected(self, event):
        self.indent_logger.log(f"Candidate rejected: {event['reason']}", header="reject")

    def on_budget_updated(self, event):
        remaining = event["metric_calls_remaining"]
        if remaining:
            self.indent_logger.log(
                f"{event['metric_calls_used']} / {event['metric_calls_used'] + remaining} calls used", header="budget"
            )


def main():
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.log("OPENROUTER_API_KEY environment variable not set", header="error")
        logger.log("Export it: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)

    # Load GPQA Diamond dataset
    logger.log("Loading GPQA Diamond dataset...", header="init")
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    # Model configuration
    task_model = "openrouter/qwen/qwen3-vl-235b-a22b-instruct"
    reflection_model = "openrouter/deepseek/deepseek-v3.2"

    # Seed prompt for GPQA multiple-choice QA
    seed_prompt = {
        "system_prompt": (
            "You are an expert scientist answering graduate-level multiple-choice questions "
            "in physics, chemistry, and biology. Analyze each question carefully, "
            "reason through the problem step by step, then provide your final answer "
            r"in the format \box{N} where N is the letter (A, B, C, or D). "
            r"For example: \box{A}"
        )
    }

    logger.log(f"Task LM: {task_model}", header="config")
    logger.log(f"Reflection LM: {reflection_model}", header="config")
    logger.show(seed_prompt["system_prompt"], title="Seed Prompt")

    # Run GEPA optimization
    result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        task_lm=task_model,
        reflection_lm=reflection_model,
        max_metric_calls=2000,
        reflection_minibatch_size=5,
        display_progress_bar=True,
        seed=42,
        litellm_cache=True,
        use_weave=True,
        weave_project_name="gpqa-diamond-dpsk",
        callbacks=[VerboseCallback()],  # pyright: ignore[reportArgumentType]
    )

    logger.log("Optimization complete!", header="done")
    logger.log(f"Best validation score: {result.val_aggregate_scores[result.best_idx]:.2%}", header="result")
    logger.show(result.best_candidate["system_prompt"], title="Optimized Prompt")


if __name__ == "__main__":
    main()
