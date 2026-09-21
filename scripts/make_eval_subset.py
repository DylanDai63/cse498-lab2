#!/usr/bin/env python3
"""Draw a reproducible, stratified evaluation subset from a ReVA split (student addition).

Why this exists
    One Qwen evaluation of the full ReVA test split (4000 questions) takes about two hours
    on the 4x RTX 2080 Ti course server, because bf16 is emulated on Turing GPUs and the
    prompt asks for step-by-step reasoning. The assignment needs three comparable runs
    (Qwen base, Qwen fine-tuned, VILA), so all three are evaluated on ONE fixed subset.

What it guarantees
    * Stratified: every question_type keeps its share of the split (largest-remainder
      rounding), so subcategory accuracies stay meaningful.
    * Reproducible: the draw depends only on --seed and the input file.
    * Stable ids: each question's id is computed on the FULL split with the same recipe as
      prepare_reva_v2_test_set.get_qa_id and written into "qa_id". Ids synthesized from a
      question's list position would otherwise change once the list is filtered.
    * Extendable: the complement ("rest") file is written too. Evaluating it later and
      scoring both prediction sets together yields the full-split result with no
      question evaluated twice.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARE_PATH = PROJECT_ROOT / "reva_eval" / "data" / "rsvidqa" / "prepare_reva_v2_test_set.py"


def load_get_qa_id():
    """Import get_qa_id from the evaluation code so both always share one id recipe."""
    spec = importlib.util.spec_from_file_location("prepare_reva_v2_test_set", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_qa_id


def flatten(data: dict[str, Any], get_qa_id) -> list[dict[str, Any]]:
    """One record per question, with its full-split stable id and its location."""
    records = []
    for video_key, video_item in data.get("videos", {}).items():
        for category, question_types in video_item.get("mcq", {}).items():
            if not isinstance(question_types, dict):
                continue
            for question_type, qa_list in question_types.items():
                if not isinstance(qa_list, list):
                    continue
                for question_idx, qa in enumerate(qa_list):
                    records.append({
                        "stable_id": get_qa_id(qa, video_item, question_type, question_idx),
                        "video_key": video_key,
                        "category": category,
                        "question_type": question_type,
                        "qa": qa,
                    })
    return records


def allocate(counts: dict[str, int], size: int) -> dict[str, int]:
    """Proportional allocation with largest-remainder rounding; sums exactly to size."""
    total = sum(counts.values())
    exact = {name: size * count / total for name, count in counts.items()}
    quota = {name: int(value) for name, value in exact.items()}
    leftover = size - sum(quota.values())
    by_remainder = sorted(counts, key=lambda name: (-(exact[name] - quota[name]), name))
    for name in by_remainder[:leftover]:
        quota[name] += 1
    return quota


def rebuild(data: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-nest selected records into the original ReVA layout, pinning qa_id."""
    videos: dict[str, Any] = {}
    for record in records:
        source = data["videos"][record["video_key"]]
        video = videos.setdefault(
            record["video_key"],
            {**{key: value for key, value in source.items() if key != "mcq"}, "mcq": {}},
        )
        qa = dict(record["qa"])
        qa["qa_id"] = record["stable_id"]
        video["mcq"].setdefault(record["category"], {}).setdefault(record["question_type"], []).append(qa)
    return {"videos": videos}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Full ReVA split, e.g. <REVA_ROOT>/test_set.json")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "eval_subsets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    records = flatten(data, load_get_qa_id())

    ids = [record["stable_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("Stable ids are not unique on the full split; refusing to sample.")
    if not 0 < args.size < len(records):
        raise SystemExit(f"--size must be between 1 and {len(records) - 1}")

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["question_type"]].append(record)
    quota = allocate({name: len(group) for name, group in by_type.items()}, args.size)

    rng = random.Random(args.seed)
    chosen_ids: set[str] = set()
    for question_type in sorted(by_type):  # sorted: the draw must not depend on dict order
        group = sorted(by_type[question_type], key=lambda record: record["stable_id"])
        chosen_ids.update(record["stable_id"] for record in rng.sample(group, quota[question_type]))

    # File order is preserved inside both outputs, so questions of one video stay together.
    subset = [record for record in records if record["stable_id"] in chosen_ids]
    rest = [record for record in records if record["stable_id"] not in chosen_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem.replace("_set", "")
    subset_path = args.output_dir / f"{stem}_subset_{len(subset)}_seed{args.seed}.json"
    rest_path = args.output_dir / f"{stem}_rest_{len(rest)}_seed{args.seed}.json"
    subset_path.write_text(json.dumps(rebuild(data, subset), ensure_ascii=False) + "\n", encoding="utf-8")
    rest_path.write_text(json.dumps(rebuild(data, rest), ensure_ascii=False) + "\n", encoding="utf-8")

    picked = Counter(record["question_type"] for record in subset)
    print(f"Full split: {len(records)} questions, {len(data['videos'])} videos")
    print(f"Subset:     {len(subset)} questions, {len({r['video_key'] for r in subset})} videos -> {subset_path}")
    print(f"Rest:       {len(rest)} questions -> {rest_path}")
    print("\nquestion_type: subset / full")
    for question_type in sorted(by_type):
        print(f"  {question_type}: {picked[question_type]} / {len(by_type[question_type])}")


if __name__ == "__main__":
    main()
