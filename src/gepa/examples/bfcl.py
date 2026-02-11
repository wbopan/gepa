# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import math
import re
from typing import Any, Optional


def init_dataset(
    categories: Optional[list[str]] = None,
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
        categories = ["simple", "multiple"]

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
                    },
                }
            )

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

    Compares predicted function calls against ground truth using
    structural matching with acceptable value lists.
    """

    def __call__(self, data: dict[str, Any], response: str) -> dict[str, Any]:
        raw_gt = json.loads(data["additional_context"]["ground_truth_raw"])
        predicted_calls = _parse_model_response(response)

        if predicted_calls is None:
            return {
                "score": 0.0,
                "feedback": f"Could not parse function calls from response. Response excerpt: {response[-300:]}",
            }

        is_correct, details = _compare_calls(predicted_calls, raw_gt)

        if is_correct:
            return {"score": 1.0, "feedback": "Correct function call(s)."}
        return {"score": 0.0, "feedback": details}


def _parse_model_response(response: str) -> Optional[list[dict[str, Any]]]:
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


def _values_match(predicted: Any, acceptable: Any) -> bool:
    """Check if a predicted value matches an acceptable value.

    Handles numeric type coercion (int/float) and string comparison.
    """
    # Both numeric: compare with tolerance
    if isinstance(predicted, (int, float)) and isinstance(acceptable, (int, float)):
        if isinstance(predicted, float) or isinstance(acceptable, float):
            return math.isclose(float(predicted), float(acceptable), rel_tol=1e-9)
        return predicted == acceptable

    # Try numeric coercion if one is string
    if isinstance(predicted, str) and isinstance(acceptable, (int, float)):
        try:
            predicted_num = float(predicted)
            if isinstance(acceptable, int) and predicted_num == int(predicted_num):
                return int(predicted_num) == acceptable
            return math.isclose(predicted_num, float(acceptable), rel_tol=1e-9)
        except (ValueError, TypeError):
            pass

    if isinstance(acceptable, str) and isinstance(predicted, (int, float)):
        try:
            acceptable_num = float(acceptable)
            if isinstance(predicted, int) and acceptable_num == int(acceptable_num):
                return predicted == int(acceptable_num)
            return math.isclose(float(predicted), acceptable_num, rel_tol=1e-9)
        except (ValueError, TypeError):
            pass

    # String comparison (case-sensitive)
    return str(predicted) == str(acceptable)


def _compare_calls(
    predicted: list[dict[str, Any]], raw_gt: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Compare predicted function calls against ground truth.

    Ground truth format: [{"func_name": {"param": [acceptable_val1, ...]}}]
    Predicted format: [{"name": "func_name", "arguments": {"param": value}}]

    For multiple calls (parallel), matching is order-independent.
    """
    if len(predicted) != len(raw_gt):
        return False, (
            f"Expected {len(raw_gt)} function call(s), got {len(predicted)}."
        )

    # For single call, match directly
    if len(raw_gt) == 1:
        ok, detail = _compare_single_call(predicted[0], raw_gt[0])
        return ok, detail

    # For multiple calls, try all permutation-free matching (greedy)
    gt_matched = [False] * len(raw_gt)
    pred_matched = [False] * len(predicted)
    mismatch_details: list[str] = []

    for pi, pred_call in enumerate(predicted):
        found = False
        for gi, gt_call in enumerate(raw_gt):
            if gt_matched[gi]:
                continue
            ok, _ = _compare_single_call(pred_call, gt_call)
            if ok:
                gt_matched[gi] = True
                pred_matched[pi] = True
                found = True
                break
        if not found:
            mismatch_details.append(
                f"Predicted call '{pred_call.get('name', '?')}' did not match any ground truth."
            )

    if all(gt_matched):
        return True, ""

    unmatched_gt = [
        list(raw_gt[i].keys())[0] for i in range(len(raw_gt)) if not gt_matched[i]
    ]
    return False, (
        f"Unmatched ground truth calls: {unmatched_gt}. " + " ".join(mismatch_details)
    )


def _compare_single_call(
    predicted: dict[str, Any], gt_call: dict[str, Any]
) -> tuple[bool, str]:
    """Compare a single predicted call against a single ground truth call.

    gt_call format: {"func_name": {"param": [acceptable_values]}}
    """
    gt_func_name = list(gt_call.keys())[0]
    gt_params = gt_call[gt_func_name]

    pred_name = predicted.get("name", "")
    pred_args = predicted.get("arguments", {})

    if pred_name != gt_func_name:
        return False, f"Expected function '{gt_func_name}', got '{pred_name}'."

    # Check each ground truth parameter
    for param_name, acceptable_values in gt_params.items():
        is_optional = "" in acceptable_values

        if param_name not in pred_args:
            if is_optional:
                continue
            return False, (
                f"Function '{gt_func_name}': missing required parameter '{param_name}'."
            )

        pred_value = pred_args[param_name]
        non_empty_acceptable = [v for v in acceptable_values if v != ""]

        if not non_empty_acceptable:
            # Only "" is acceptable, meaning param should not be provided
            # but model provided it - that's okay if value is empty-ish
            continue

        matched = any(_values_match(pred_value, av) for av in non_empty_acceptable)
        if not matched:
            return False, (
                f"Function '{gt_func_name}': parameter '{param_name}' = {pred_value!r}, "
                f"expected one of {non_empty_acceptable}."
            )

    # Check for extra parameters not in ground truth
    for param_name in pred_args:
        if param_name not in gt_params:
            return False, (
                f"Function '{gt_func_name}': unexpected parameter '{param_name}'."
            )

    return True, ""
