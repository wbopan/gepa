#!/usr/bin/env python3
"""NYT Connections experiment with GEPA."""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.examples.nyt_connections import ConnectionsEvaluator, init_dataset

logger = get_logger()


def main() -> None:
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    LiteLLMCacheLogger.register()

    result = gepa.optimize(
        seed_candidate={
            "system_prompt": (
                "You are an expert at solving NYT Connections puzzles. "
                "Given 16 words, identify four groups of four words that share a common theme or connection.\n\n"
                "Output exactly four lines. Each line should contain four comma-separated words forming one group. "
                "No group names, explanations, or other text - just the four lines of grouped words.\n\n"
                "Think carefully about:\n"
                "- Literal categories (same type of thing)\n"
                "- Wordplay (homophones, palindromes, letter patterns)\n"
                "- Cultural references (phrases, titles, names)\n"
                "- Tricky connections that seem to fit multiple categories"
            )
        },
        trainset=trainset,
        valset=valset,
        task_lm="openrouter/deepseek/deepseek-v3.2",
        task_lm_kwargs={"max_tokens": 1000},
        evaluator=ConnectionsEvaluator(),
        reflection_lm="openrouter/deepseek/deepseek-v3.2",
        max_metric_calls=2000,
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
