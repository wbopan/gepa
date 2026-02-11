#!/usr/bin/env python3
"""GPQA Diamond experiment using MemoryAdapter with DeepSeek V3.2."""

import gepa
from gepa import LiteLLMCacheLogger, VerboseCallback, get_logger
from gepa.adapters.memory_adapter import MemoryAdapter
from gepa.examples.gpqa import init_dataset

logger = get_logger()

TASK_MODEL = "openrouter/deepseek/deepseek-v3.2"
REFLECTION_MODEL = "openrouter/deepseek/deepseek-v3.2"

SYSTEM_PROMPT = (
    "You are an expert scientist answering graduate-level "
    "multiple-choice questions in physics, chemistry, and biology. "
    "Analyze each question carefully, reason through the problem "
    r"step by step, then provide your final answer in the format \boxed{N} "
    r"where N is the letter (A, B, C, or D). For example: \boxed{A}"
)


def gpqa_evaluator(data, response):
    """Check if the expected boxed answer appears in the response."""
    expected = data["answer"]  # e.g. r"\boxed{A}"
    if expected in response:
        return {"score": 1.0, "feedback": f"Correct. Expected {expected} found in response."}
    return {
        "score": 0.0,
        "feedback": f"Incorrect. Expected {expected} but not found in response. Response excerpt: {response[-200:]}",
    }


def main() -> None:
    trainset, valset, _ = init_dataset()
    trainset, valset = trainset[:30], valset[:30]
    logger.log(f"Loaded {len(trainset)} train, {len(valset)} val examples", header="init")

    LiteLLMCacheLogger.register()

    adapter = MemoryAdapter(
        task_model=TASK_MODEL,
        reflection_model=REFLECTION_MODEL,
        evaluator=gpqa_evaluator,
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
