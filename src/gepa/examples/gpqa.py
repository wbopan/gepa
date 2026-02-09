# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


def init_dataset():
    """
    Load GPQA Diamond dataset.

    GPQA is a Graduate-Level Google-Proof Q&A benchmark with very hard
    multiple-choice questions in biology, physics, and chemistry.

    Returns:
        trainset: First 98 examples
        valset: Last 100 examples
        testset: Empty list (no separate test set)
    """
    from datasets import load_dataset

    # Load GPQA Diamond subset
    # Note: You may need to authenticate with HuggingFace and agree to dataset terms
    # Run: huggingface-cli login
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]

    def format_example(x):
        # Format the question with answer choices
        import hashlib
        import random

        question = x["Question"]
        answers = [
            x["Correct Answer"],
            x["Incorrect Answer 1"],
            x["Incorrect Answer 2"],
            x["Incorrect Answer 3"],
        ]
        # Shuffle answers to avoid correct answer always being first
        # Use hashlib instead of hash() for deterministic hashing across sessions
        question_hash = int(hashlib.md5(question.encode()).hexdigest(), 16) % (2**32)
        random.seed(question_hash)
        random.shuffle(answers)

        # Find the correct answer letter after shuffling
        correct_answer = x["Correct Answer"]
        correct_letter = r"\boxed{" + ["A", "B", "C", "D"][answers.index(correct_answer)] + "}"

        # Format choices with proper A, B, C, D labels
        choices = [f"{letter}) {ans}" for letter, ans in zip(["A", "B", "C", "D"], answers, strict=True)]
        formatted_input = f"{question}\n\n" + "\n".join(choices)

        return {
            "input": formatted_input,
            "answer": correct_letter,
            "additional_context": {
                "domain": x.get("High-level domain", ""),
                "subdomain": x.get("Subdomain", ""),
            }
        }

    all_examples = [format_example(x) for x in dataset]

    # First 98 as training set, last 100 as validation set
    trainset = all_examples[:98]
    valset = all_examples[-100:]

    return trainset, valset, []
