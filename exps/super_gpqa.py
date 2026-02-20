#!/usr/bin/env python3
"""SuperGPQA experiment with GEPA."""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.examples.super_gpqa import init_dataset

logger = get_logger()


def main() -> None:
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    LiteLLMCacheLogger.register()

    result = gepa.optimize(
        seed_candidate={
            "system_prompt": (
                "You are an expert scholar answering graduate-level "
                "multiple-choice questions spanning 285 disciplines. "
                "Each question has a variable number of options (up to 10, labeled A-J). "
                "Analyze each question carefully, reason through the problem "
                r"step by step, then provide your final answer in the format \boxed{N} "
                r"where N is the letter of the correct option. For example: \boxed{A}"
            )
        },
        trainset=trainset,
        valset=valset,
        task_lm="openrouter/openai/gpt-5-mini",
        task_lm_kwargs={"max_tokens": 8000},
        reflection_lm="openrouter/deepseek/deepseek-v3.2",
        max_metric_calls=1000,
        reflection_minibatch_size=5,
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
