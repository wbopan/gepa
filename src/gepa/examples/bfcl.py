# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import random
import re
from typing import Any


def init_dataset(
    categories: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load BFCL v3 dataset for function-calling evaluation.

    Uses hf_hub_download (not load_dataset) per BFCL guidelines.
    Downloads question and answer JSONL files, joins by ID.

    Args:
        categories: Which BFCL categories to include. Defaults to ["simple", "multiple"].

    Returns:
        trainset: First half of examples
        valset: Second half of examples
        testset: Empty list
    """
    from huggingface_hub import hf_hub_download

    if categories is None:
        categories = ["live_simple"]

    repo_id = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

    all_examples: list[dict[str, Any]] = []

    for category in categories:
        question_file = hf_hub_download(
            repo_id=repo_id,
            filename=f"BFCL_v3_{category}.json",
            repo_type="dataset",
        )
        answer_file = hf_hub_download(
            repo_id=repo_id,
            filename=f"possible_answer/BFCL_v3_{category}.json",
            repo_type="dataset",
        )

        # Parse JSONL files
        questions: dict[str, dict[str, Any]] = {}
        with open(question_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                questions[entry["id"]] = entry

        answers: dict[str, Any] = {}
        with open(answer_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                answers[entry["id"]] = entry

        # Join by ID and format
        for qid, q in questions.items():
            if qid not in answers:
                continue

            raw_ground_truth = answers[qid]["ground_truth"]

            # Extract user question: double-nested list [[{"role": "user", "content": "..."}]]
            user_content = q["question"][0][0]["content"]
            function_schemas = q["function"]

            # Build canonical answer string for feedback display
            canonical_calls = _ground_truth_to_canonical(raw_ground_truth)
            answer_str = json.dumps(canonical_calls)

            formatted_input = (
                f"Question: {user_content}\n\n"
                f"Available Functions:\n{json.dumps(function_schemas, indent=2)}\n\n"
                "Call the appropriate function(s). Output ONLY a JSON list of function calls:\n"
                '[{"name": "func_name", "arguments": {"key": "value"}}]'
            )

            all_examples.append(
                {
                    "input": formatted_input,
                    "answer": answer_str,
                    "additional_context": {
                        "category": category,
                        "id": qid,
                        "ground_truth_raw": json.dumps(raw_ground_truth),
                        "function_schemas": json.dumps(function_schemas),
                    },
                }
            )

    # Shuffle deterministically before splitting to ensure category balance
    random.Random(0).shuffle(all_examples)

    # Split: first half train, second half val
    mid = len(all_examples) // 2
    trainset = all_examples[:mid]
    valset = all_examples[mid:]

    return trainset, valset, []


def _ground_truth_to_canonical(raw_gt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ground truth (with acceptable-value lists) to a canonical call list.

    Takes the first acceptable value for each parameter. Used for the `answer` field
    (human-readable feedback), not for evaluation.
    """
    calls = []
    for call_gt in raw_gt:
        for func_name, params in call_gt.items():
            arguments = {}
            for param_name, acceptable_values in params.items():
                # Skip optional params where "" is the only acceptable value
                if acceptable_values == [""]:
                    continue
                # Take first non-empty value
                for v in acceptable_values:
                    if v != "":
                        arguments[param_name] = v
                        break
            calls.append({"name": func_name, "arguments": arguments})
    return calls


class BFCLEvaluator:
    """Evaluator for BFCL function-calling tasks.

    Uses the official bfcl-eval ast_checker for evaluation.
    """

    def __call__(self, data: dict[str, Any], response: str) -> dict[str, Any]:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

        category = data["additional_context"]["category"]
        raw_gt = json.loads(data["additional_context"]["ground_truth_raw"])
        func_schemas = json.loads(data["additional_context"]["function_schemas"])

        parsed = _parse_model_response(response)
        if parsed is None:
            return {
                "score": 0.0,
                "feedback": f"Could not parse function calls from response. Response excerpt: {response[-300:]}",
            }

        # Convert our format [{"name": "f", "arguments": {...}}] to official format [{"f": {...}}]
        decoded = [{call["name"]: call.get("arguments", {})} for call in parsed]

        result = ast_checker(func_schemas, decoded, raw_gt, Language.PYTHON, category, "DeepSeek-V3.2-Exp")

        if result["valid"]:
            return {"score": 1.0, "feedback": "Correct function call(s)."}
        error_msg = "; ".join(result.get("error", []))
        return {"score": 0.0, "feedback": error_msg or "Incorrect function call."}


def _parse_model_response(response: str) -> list[dict[str, Any]] | None:
    """Parse model response into a list of function call dicts.

    Supports:
    - Raw JSON list: [{"name": ..., "arguments": {...}}]
    - Markdown code block: ```json ... ```
    """
    response = response.strip()

    # Try extracting from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if code_block_match:
        response = code_block_match.group(1).strip()

    # Find the outermost JSON array
    start = response.find("[")
    end = response.rfind("]")
    if start != -1 and end != -1 and end > start:
        response = response[start : end + 1]

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        parsed = [parsed]

    # Validate structure
    for call in parsed:
        if not isinstance(call, dict):
            return None
        if "name" not in call:
            return None
        if "arguments" not in call:
            call["arguments"] = {}

    return parsed
