#!/usr/bin/env python3
"""GPQA Diamond experiment with GEPA using PMaxBatchSampler."""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.examples.gpqa import init_dataset
from gepa.strategies import PMaxBatchSampler

logger = get_logger()


def main() -> None:
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
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
        task_lm="openrouter/qwen/qwen3-vl-235b-a22b-instruct",
        reflection_lm="openrouter/deepseek/deepseek-v3.2",
        max_metric_calls=2000,
        batch_sampler=PMaxBatchSampler(minibatch_size=5),
        display_progress_bar=True,
        seed=42,
        use_weave=True,
        weave_project_name="gepa-boost",
        callbacks=[VerboseCallback()],  # pyright: ignore[reportArgumentType]
    )

    logger.log("Optimization complete!", header="done")
    logger.log(f"Best validation score: {result.val_aggregate_scores[result.best_idx]:.2%}", header="result")
    logger.show(result.best_candidate["system_prompt"], title="Optimized Prompt")


if __name__ == "__main__":
    main()
