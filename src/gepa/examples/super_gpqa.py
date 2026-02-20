# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


def init_dataset(
    discipline: str | None = None,
    difficulty: str | None = None,
    n_train: int = 100,
    n_val: int = 100,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Load SuperGPQA dataset.

    SuperGPQA is a graduate-level QA benchmark with 26,500 questions across
    285 disciplines. Each question has 4-10 multiple-choice options.

    Args:
        discipline: Filter by discipline (e.g. "Science", "Engineering").
        difficulty: Filter by difficulty ("easy", "middle", "hard").
        n_train: Number of training examples to sample.
        n_val: Number of validation examples to sample.
        seed: Random seed for reproducible train/val split.

    Returns:
        trainset: Sampled training examples.
        valset: Sampled validation examples.
        testset: Empty list (no separate test set).
    """
    import hashlib
    import random
    import string

    from datasets import load_dataset

    dataset = load_dataset("m-a-p/SuperGPQA", split="train")

    # Optional filtering
    if discipline is not None:
        dataset = dataset.filter(lambda x: x["discipline"] == discipline)
    if difficulty is not None:
        dataset = dataset.filter(lambda x: x["difficulty"] == difficulty)

    def format_example(x):
        question = x["question"]
        options = list(x["options"])
        correct_answer = x["answer"]

        # Shuffle options deterministically using MD5 hash of the question
        question_hash = int(hashlib.md5(question.encode()).hexdigest(), 16) % (2**32)
        random.seed(question_hash)
        random.shuffle(options)

        # Generate letter labels for however many options there are (A-J)
        n_options = len(options)
        letters = list(string.ascii_uppercase[:n_options])

        correct_letter = r"\boxed{" + letters[options.index(correct_answer)] + "}"

        choices = [f"{letter}) {ans}" for letter, ans in zip(letters, options, strict=True)]
        formatted_input = f"{question}\n\n" + "\n".join(choices)

        return {
            "input": formatted_input,
            "answer": correct_letter,
            "additional_context": {
                "discipline": x["discipline"],
                "field": x["field"],
                "subfield": x["subfield"],
                "difficulty": x["difficulty"],
            },
        }

    all_examples = [format_example(x) for x in dataset]

    # Deterministic sampling for train/val split
    rng = random.Random(seed)
    total = min(n_train + n_val, len(all_examples))
    sampled = rng.sample(all_examples, total)
    split = min(n_train, total * n_train // (n_train + n_val)) if total < n_train + n_val else n_train
    trainset = sampled[:split]
    valset = sampled[split:]

    return trainset, valset, []
