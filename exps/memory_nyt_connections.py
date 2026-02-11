#!/usr/bin/env python3
"""NYT Connections experiment using MemoryAdapter with DeepSeek V3.2."""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.adapters.memory_adapter import MemoryAdapter
from gepa.examples.nyt_connections import ConnectionsEvaluator, init_dataset

logger = get_logger()

TASK_MODEL = "openrouter/deepseek/deepseek-v3.2"
REFLECTION_MODEL = "openrouter/deepseek/deepseek-v3.2"

SYSTEM_PROMPT = (
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


class MemoryConnectionsEvaluator:
    """Wraps ConnectionsEvaluator to return MemoryAdapter-compatible EvaluationResult."""

    def __init__(self):
        self._inner = ConnectionsEvaluator()

    def __call__(self, data, response):
        result = self._inner(data, response)
        return {"score": result.score, "feedback": result.feedback}


def main() -> None:
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    LiteLLMCacheLogger.register()

    adapter = MemoryAdapter(
        task_model=TASK_MODEL,
        reflection_model=REFLECTION_MODEL,
        evaluator=MemoryConnectionsEvaluator(),
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
        reflection_minibatch_size=5,
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
