#!/usr/bin/env python3
"""BFCL v3 function-calling experiment using MemoryAdapter with AdaBoost batch sampling."""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.adapters.memory_adapter import MemoryAdapter
from gepa.examples.bfcl import BFCLEvaluator, init_dataset
from gepa.strategies import AdaBoostBatchSampler

logger = get_logger()

TASK_MODEL = "openrouter/deepseek/deepseek-v3.2"
REFLECTION_MODEL = "openrouter/deepseek/deepseek-v3.2"

SYSTEM_PROMPT = (
    "You are a function calling AI model. You are provided with function signatures "
    "within the query. Call the appropriate function(s) to answer the user's query.\n\n"
    "Output a JSON list of function calls:\n"
    '[{"name": "function_name", "arguments": {"param": "value"}}]\n\n'
    "Rules:\n"
    "- Match parameter types exactly (int, float, string, bool)\n"
    "- Include all required parameters\n"
    "- Output ONLY the JSON, no explanation"
)


def main() -> None:
    trainset, valset, _ = init_dataset(
        categories=["live_simple"]
    )
    trainset, valset = trainset[:30], valset[:30]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    LiteLLMCacheLogger.register()

    adapter = MemoryAdapter(
        task_model=TASK_MODEL,
        reflection_model=REFLECTION_MODEL,
        evaluator=BFCLEvaluator(),
        base_system_prompt=SYSTEM_PROMPT,
        max_entries=50,
        max_retries=2,
    )

    result = gepa.optimize(
        seed_candidate={"memory": '<memory>\n<entry key="General Knowledge">Think step by step carefully.</entry>\n</memory>'},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=1000,
        batch_sampler=AdaBoostBatchSampler(minibatch_size=5),
        display_progress_bar=True,
        seed=42,
        use_weave=True,
        weave_project_name="gepa-boost",
        callbacks=[VerboseCallback()],  # pyright: ignore[reportArgumentType]
    )

    logger.log("Optimization complete!", header="done")
    logger.log(f"Best validation score: {result.val_aggregate_scores[result.best_idx]:.2%}", header="result")
    logger.show(result.best_candidate["memory"], title="Optimized Memory")


if __name__ == "__main__":
    main()
