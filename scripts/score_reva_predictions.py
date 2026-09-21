#!/usr/bin/env python3
"""Score Qwen-style ReVA prediction JSONL files.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_FLAGS = re.DOTALL | re.IGNORECASE


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    # TODO(student): use regex with DOTALL, strip whitespace, and return a string.
    # DOTALL lets "." cross newlines (models often put the letter on its own line).
    # The LAST non-empty tag wins: a model may mention "<answer>" while reasoning and
    # only commit to its final choice at the end.
    text = "" if text is None else str(text)
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=_FLAGS)
    for inner in reversed(matches):
        if inner.strip():
            return inner.strip()
    if matches:
        return ""
    return text.strip()


def _parse_letter(candidate: str) -> str:
    """Parse one option letter from an already-isolated answer string ("" if unclear).

    Tiers go from most to least explicit. When the text is ambiguous the function
    returns "" on purpose: an unparsable prediction is scored as incorrect, which is
    honest, whereas guessing a letter would inflate or deflate accuracy at random.
    """
    # Tier 1 - the whole string is one letter, maybe wrapped: "b", "(B)", "B.", "**C**".
    match = re.fullmatch(r"[\s\(\[\"'*]*([A-Ha-h])[\s\)\]\.\:\"'*]*", candidate)
    if match:
        return match.group(1).upper()

    # Tier 2 - the string starts with a letter label: "B. a video clip", "(b) video".
    match = re.match(r"[\s\(\[*]*([A-H])\s*[\)\]\.\:,\-]", candidate) or re.match(r"\s*\(?([a-h])\)", candidate)
    if match:
        return match.group(1).upper()

    # Tier 3 - an explicit cue: "The answer is C because...", "Final answer: C".
    # The letter must be upper-case here so "the answer is a video" is not read as A.
    cues = re.findall(
        r"(?i:\banswer\b)(?i:\s+(?:is|would\s+be|should\s+be))?\s*[:\-=]?\s*\*{0,2}\(?([A-H])\b",
        candidate,
    )
    if cues:
        return cues[-1]

    # Tier 4 - exactly one distinct stand-alone capital letter in the whole text.
    # "A" followed by a lower-case word is the English article ("A car turns left").
    without_article = re.sub(r"\bA(?=\s+[a-z])", " ", candidate)
    letters = set(re.findall(r"\b([A-H])\b", without_article))
    if len(letters) == 1:
        return letters.pop()
    return ""


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    # TODO(student): normalize case and robustly parse A-H from short or verbose output.
    raw = "" if text is None else str(text)
    candidate = extract_answer(raw)
    if not re.search(r"<answer>.*?</answer>", raw, flags=_FLAGS):
        # No answer tag: the fallback is the full output, so drop reasoning blocks first.
        # Letters named inside <think> ("A and B are distractors") must never be parsed.
        candidate = re.sub(r"<think>.*?</think>", " ", candidate, flags=_FLAGS)
        candidate = re.sub(r"<think>.*", " ", candidate, flags=_FLAGS)  # unclosed block
        candidate = candidate.strip()
    return _parse_letter(candidate)


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    # TODO(student): read every prediction *.json line-by-line and index by item["id"].
    # Inference writes one JSON object per line: pred.json for a single GPU, or one
    # shard per GPU ("4_0.json" ... "4_3.json") when NUM_CHUNKS > 1. result.json is this
    # scorer's own output and must not be read back as predictions.
    preds: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(str(output_dir), "*.json"))):
        if os.path.basename(path) == "result.json":
            continue
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # A partially written last line (interrupted run) should not abort scoring.
                    print(f"Warning: skipped malformed line {line_number} in {path}")
                    continue
                if isinstance(item, dict) and "id" in item:
                    preds[item["id"]] = item
    return preds


def _ratio_line(label: str, numerator: int, denominator: int) -> str:
    percentage = 100.0 * numerator / denominator if denominator else 0.0
    return f"{label}: {numerator}/{denominator} = {percentage:.2f}%"


def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    # TODO(student): compare predictions against every GT item, count missing predictions
    # as incorrect, and report Completed plus Total/subcategory accuracy statistics.
    # The loop runs over the GROUND TRUTH, not over the predictions, so the denominator
    # is always the full question set. A model that answers 10 of 100 questions and gets
    # 9 right scores 9%, not 90%.
    results: dict[str, dict[str, Any]] = {}
    total = len(gt_list)
    completed = 0
    missing = 0
    correct = 0
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # question_type -> [correct, total]

    for gt in gt_list:
        qa_id = gt["id"]
        question_type = gt.get("question_type", "unknown")
        gt_letter = str(gt.get("answer", "")).strip().upper()

        pred = preds.get(qa_id)
        if pred is None:
            missing += 1
            pred_text = ""
            pred_letter = ""
        else:
            pred_text = str(pred.get("pred", ""))
            pred_letter = extract_letter(pred_text)

        # "Completed" = a prediction exists AND a letter could be parsed from it. This
        # matches num_answered in vila_eval/reva_v2.py, so the two models are comparable.
        if pred_letter:
            completed += 1
        acc = int(bool(pred_letter) and pred_letter == gt_letter)
        correct += acc
        per_type[question_type][0] += acc
        per_type[question_type][1] += 1

        results[qa_id] = {
            "question_type": question_type,
            "gt": gt_letter,
            "pred": pred_text,
            "pred_letter": pred_letter,
            "missing": pred is None,
            "acc": acc,
        }

    # "Completed:" and "Total:" are parsed line-by-line by scripts/compare_model_metrics.py,
    # so their format is fixed: "<label>: <num>/<den> = <pct>%".
    lines = [
        _ratio_line("Completed", completed, total),
        _ratio_line("Total", correct, total),
        f"Missing predictions: {missing}",
        f"Unparsable predictions: {total - missing - completed}",
    ]
    for question_type in sorted(per_type):
        type_correct, type_total = per_type[question_type]
        lines.append(_ratio_line(question_type, type_correct, type_total))
    return results, "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = load_predictions(args.output_dir)
    gt_list = json.loads(args.gt_file.read_text(encoding="utf-8"))
    results, csv_text = score_predictions(preds, gt_list)

    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Results ===")
    print(csv_text)

    csv_path = args.output_dir / "result.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
