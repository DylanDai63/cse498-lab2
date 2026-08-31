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


def extract_answer(text: str) -> str:
    """Extract text inside <answer>...</answer>; fall back to full text."""
    # TODO(student): use regex with DOTALL, strip whitespace, and return a string.
    raise NotImplementedError("TODO: implement extract_answer")


def extract_letter(text: str) -> str:
    """Extract one answer letter A-H from model output."""
    # TODO(student): normalize case and robustly parse A-H from short or verbose output.
    raise NotImplementedError("TODO: implement extract_letter")


def load_predictions(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load prediction JSONL files from output_dir, ignoring result.json."""
    # TODO(student): read every prediction *.json line-by-line and index by item["id"].
    raise NotImplementedError("TODO: implement load_predictions")


def score_predictions(preds: dict[str, dict[str, Any]], gt_list: list[dict[str, Any]]) -> tuple[dict, str]:
    """Score every GT item and report completion separately from accuracy."""
    # TODO(student): compare predictions against every GT item, count missing predictions
    # as incorrect, and report Completed plus Total/subcategory accuracy statistics.
    raise NotImplementedError("TODO: implement score_predictions")


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
