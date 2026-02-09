#!/usr/bin/env python3
"""NYT Connections experiment with GEPA - Test version."""

import gepa
from gepa import LiteLLMCacheLogger, get_logger
from gepa.examples.nyt_connections import ConnectionsEvaluator, init_dataset

logger = get_logger()


def main() -> None:
    # Load dataset with small subset for testing
    trainset, valset, _ = init_dataset()
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples total", header="init")

    # Use only 2 examples each for quick testing
    trainset, valset = trainset[:2], valset[:2]
    logger.log(f"Using {len(trainset)} train, {len(valset)} val examples for test", header="init")

    # Print sample data for verification
    logger.log("Sample training example:", header="sample")
    sample = trainset[0]
    print(f"\nInput:\n{sample['input']}")
    print(f"\nExpected Answer:\n{sample['answer']}")
    print(f"\nAdditional Context: {sample['additional_context']}\n")

    LiteLLMCacheLogger.register()

    # Custom evaluator for Connections scoring
    evaluator = ConnectionsEvaluator()

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
        task_lm="openrouter/openai/gpt-4o-mini",
        task_lm_kwargs={"max_tokens": 500},
        evaluator=evaluator,
        reflection_lm="openrouter/deepseek/deepseek-v3.2",
        max_metric_calls=4,  # Just one evaluation round
        reflection_minibatch_size=2,
        display_progress_bar=True,
        seed=42,
        use_weave=False,  # Disable for testing
        track_best_outputs=True,
    )

    logger.log("Test complete!", header="done")
    logger.log(f"Validation scores: {result.val_aggregate_scores}", header="result")

    # Print outputs for verification
    logger.log("Sample outputs:", header="output")
    if result.best_outputs_valset:
        for val_id, outputs in result.best_outputs_valset.items():
            if outputs:
                _, output = outputs[0]  # Get first best output
                print(f"\n--- Val ID: {val_id} ---")
                print(f"Input: {valset[val_id]['input'][:100]}...")
                print(f"Model output:\n{output.get('full_assistant_response', 'N/A')}")
                print(f"Expected:\n{valset[val_id]['answer']}")


if __name__ == "__main__":
    main()
