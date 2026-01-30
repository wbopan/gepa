#!/usr/bin/env python3
"""
GPQA Diamond experiment with GEPA using OpenRouter + DeepSeek V3.2.
Run from project root: uv run run.py
"""

import logging
import os
import sys
import warnings

import litellm
from litellm.integrations.custom_logger import CustomLogger

# Suppress noisy output
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
litellm.suppress_debug_info = True


class LiteLLMCacheLogger(CustomLogger):
    """Logs LiteLLM request timing and cache hits/misses."""

    def log_pre_api_call(self, model, messages, kwargs):
        """Called before each API request is sent."""
        from datetime import datetime
        print(f"  [{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [REQUEST START] {model}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Called when a request completes successfully."""
        cache_hit = kwargs.get("cache_hit", False)
        model = kwargs.get("model", "unknown")
        duration = (end_time - start_time).total_seconds()
        status = "CACHE HIT" if cache_hit else "CACHE MISS"
        print(f"  [{end_time.strftime('%H:%M:%S.%f')[:-3]}] [{status}] {model} ({duration:.2f}s)")


# Register litellm callback
litellm.callbacks = [LiteLLMCacheLogger()]

import gepa
from gepa.examples.gpqa import init_dataset


class VerboseCallback:
    """Prints detailed progress during optimization."""

    def on_optimization_start(self, event):
        print(f"[START] Optimization started: {event['trainset_size']} train, {event['valset_size']} val examples")

    def on_iteration_start(self, event):
        print(f"\n[ITER {event['iteration']}] Starting iteration...")

    def on_candidate_selected(self, event):
        print(f"  [SELECT] Candidate {event['candidate_idx']} selected (score: {event['score']:.2%})")

    def on_minibatch_sampled(self, event):
        print(f"  [SAMPLE] Minibatch of {len(event['minibatch_ids'])} examples sampled")

    def on_evaluation_start(self, event):
        print(f"  [EVAL] Evaluating batch of {event['batch_size']} examples...")

    def on_evaluation_end(self, event):
        avg = sum(event['scores']) / len(event['scores']) if event['scores'] else 0
        print(f"  [EVAL] Done. Avg score: {avg:.2%}")

    def on_candidate_accepted(self, event):
        print(f"  [ACCEPT] New candidate {event['new_candidate_idx']} accepted (score: {event['new_score']:.2f})")

    def on_candidate_rejected(self, event):
        print(f"  [REJECT] Candidate rejected: {event['reason']}")

    def on_budget_updated(self, event):
        remaining = event['metric_calls_remaining']
        if remaining:
            print(f"  [BUDGET] {event['metric_calls_used']} / {event['metric_calls_used'] + remaining} calls used")


def main():
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Export it: export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)

    # Load GPQA Diamond dataset
    print("Loading GPQA Diamond dataset...")
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
    print(f"Loaded {len(trainset)} train, {len(valset)} val examples")

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

    print(f"Starting GEPA optimization with task_lm: {task_model}, reflection_lm: {reflection_model}")
    print("-" * 50)

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
        use_mlflow=True,
        mlflow_experiment_name="gpqa-diamond-dpsk",
        callbacks=[VerboseCallback()],  # pyright: ignore[reportArgumentType]
    )

    print("-" * 50)
    print("Optimization complete!")
    print(f"\nBest validation score: {result.val_aggregate_scores[result.best_idx]:.2%}")
    print(f"\nOptimized prompt:\n{result.best_candidate['system_prompt']}")


if __name__ == "__main__":
    main()
