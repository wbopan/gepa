# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import hashlib
import random
from typing import Any

from gepa.adapters.default_adapter.default_adapter import DefaultDataInst, EvaluationResult


class ConnectionsEvaluator:
    """Evaluator for NYT Connections puzzles.

    Scores based on number of correctly grouped categories (0, 0.25, 0.5, 0.75, or 1.0).
    """

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        # Parse expected answer into groups
        answer_groups = self._parse_groups(data["answer"])

        # Parse model response into groups
        predicted_groups = self._parse_groups(response)

        # Score the prediction
        correct_count = self._count_correct_groups(predicted_groups, answer_groups)
        score = correct_count / 4

        # Generate feedback
        if correct_count == 4:
            feedback = "All 4 groups correctly identified."
        else:
            feedback = self._generate_feedback(predicted_groups, answer_groups, data)

        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)

    def _parse_groups(self, text: str) -> list[set[str]]:
        """Parse text into groups of words."""
        groups = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Split by comma and clean up
            words = [w.strip().upper() for w in line.split(",") if w.strip()]
            if words:
                groups.append(set(words))
        return groups

    def _count_correct_groups(
        self, predicted: list[set[str]], answer: list[set[str]]
    ) -> int:
        """Count how many predicted groups exactly match answer groups."""
        correct = 0
        used_answers = set()
        for pred in predicted:
            for i, ans in enumerate(answer):
                if i in used_answers:
                    continue
                if pred == ans:
                    correct += 1
                    used_answers.add(i)
                    break
        return correct

    def _generate_feedback(
        self,
        predicted: list[set[str]],
        answer: list[set[str]],
        data: DefaultDataInst,
    ) -> str:
        """Generate detailed feedback about incorrect groups."""
        correct_groups = []
        incorrect_groups = []
        used_answers: set[int] = set()

        for pred in predicted:
            matched = False
            for i, ans in enumerate(answer):
                if i in used_answers:
                    continue
                if pred == ans:
                    correct_groups.append(pred)
                    used_answers.add(i)
                    matched = True
                    break
            if not matched:
                incorrect_groups.append(pred)

        feedback_parts = []
        if correct_groups:
            feedback_parts.append(
                f"Correctly grouped: {len(correct_groups)}/4 categories."
            )
        if incorrect_groups:
            feedback_parts.append(f"Incorrectly grouped: {len(incorrect_groups)} categories.")

        # Add category hints from additional_context if available
        categories = data.get("additional_context", {}).get("categories", "")
        if categories:
            feedback_parts.append(f"The actual categories were: {categories}")

        return " ".join(feedback_parts)


def _example_to_data_inst(example: dict[str, Any]) -> DefaultDataInst:
    """Convert HuggingFace example to GEPA DataInst format."""
    words = list(example["words"])
    answers = example["answers"]
    difficulty = example.get("difficulty") or 0
    date_str = str(example.get("date") or "")

    # Shuffle words deterministically using date
    date_hash = int(hashlib.md5(date_str.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(date_hash)
    rng.shuffle(words)

    # Format input
    input_text = "Words: " + ", ".join(words)

    # Format answer (groups in order)
    answer_lines = []
    categories = []
    for answer_group in answers:
        group_words = answer_group["words"]
        answer_lines.append(", ".join(group_words))
        categories.append(answer_group.get("answerDescription", ""))
    answer_text = "\n".join(answer_lines)

    return {
        "input": input_text,
        "answer": answer_text,
        "additional_context": {
            "categories": " | ".join(categories),
            "difficulty": f"{difficulty:.1f}",
            "date": date_str,
        },
    }


def init_dataset() -> tuple[list[DefaultDataInst], list[DefaultDataInst], list[DefaultDataInst]]:
    """
    Load NYT Connections dataset from HuggingFace.

    Uses the tm21cy/NYT-Connections dataset which contains 650+ puzzles.

    Returns:
        trainset: First half of examples
        valset: Second half of examples
        testset: Empty list
    """
    from datasets import load_dataset

    dataset = load_dataset("tm21cy/NYT-Connections")["train"]

    all_examples: list[DefaultDataInst] = []
    for example in dataset:
        data_inst = _example_to_data_inst(example)
        all_examples.append(data_inst)

    # Split into train/val (50/50)
    mid_point = len(all_examples) // 2
    trainset = all_examples[:mid_point]
    valset = all_examples[mid_point:]

    return trainset, valset, []
