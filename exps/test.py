#!/usr/bin/env python3
"""Minimal e2e test for GEPA with cache verification.

A lightweight version of gpqa_diamond.py for quick validation.
- Uses tiny dataset (3 train, 3 val)
- Disables wandb/weave
- Low metric call budget (50)
"""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.examples.gpqa import init_dataset

logger = get_logger()


def main() -> None:
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:3], valset[:3]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    LiteLLMCacheLogger.register()

    result = gepa.optimize(
        seed_candidate={
            "system_prompt": (
                "You are an expert scientist answering graduate-level "
                "multiple-choice questions in physics, chemistry, and biology. "
                "Analyze each question carefully, reason through the problem "
                r"step by step, then provide your final answer in the format \box{N} "
                r"where N is the letter (A, B, C, or D). For example: \box{A}"
            )
        },
        trainset=trainset,
        valset=valset,
        task_lm="openrouter/openai/gpt-4o-mini",
        reflection_lm="openrouter/openai/gpt-4o-mini",
        max_metric_calls=3,  # Only run initial valset eval (3 examples) for cache testing
        reflection_minibatch_size=3,
        display_progress_bar=True,
        seed=42,
        use_weave=False,
        callbacks=[VerboseCallback()],  # pyright: ignore[reportArgumentType]
    )

    logger.log("Optimization complete!", header="done")
    logger.log(f"Best validation score: {result.val_aggregate_scores[result.best_idx]:.2%}", header="result")
    logger.show(result.best_candidate["system_prompt"], title="Optimized Prompt")


if __name__ == "__main__":
    main()
