# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


def init_dataset():
    from datasets import load_dataset

    from gepa.datasets import split_and_shuffle

    train_split = [
        {"input": x["problem"], "additional_context": {"solution": x["solution"]}, "answer": "### " + str(x["answer"])}
        for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
    ]
    test_split = [
        {"input": x["problem"], "answer": "### " + str(x["answer"])}
        for x in load_dataset("MathArena/aime_2025")["train"]
    ]

    trainset, valset = split_and_shuffle(train_split, seed=0)
    testset = test_split * 5

    return trainset, valset, testset
