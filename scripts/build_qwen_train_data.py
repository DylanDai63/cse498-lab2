#!/usr/bin/env python3
"""Convert ReVA annotations into Qwen-VL conversation training data.

Student task: complete every block marked TODO(student).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_video_path(file_path: str, video_root: Path | None = None) -> str:
    path = Path(file_path)
    if video_root is None:
        return path.as_posix()
    return path.relative_to(video_root).as_posix() if path.is_absolute() else path.as_posix()


def format_options(options: dict[str, str]) -> str:
    """Return sorted multiple-choice options, one option per line."""
    # TODO(student): sort option labels and format each line as "A. option text".
    raise NotImplementedError("TODO: implement format_options")


def format_question(question: str, options: dict[str, str], prompt_style: str) -> str:
    """Build the human message used by Qwen SFT."""
    # TODO(student): call format_options and support "plain" and "reva_eval" styles.
    raise NotImplementedError("TODO: implement format_question")


def format_answer(qa: dict[str, Any], answer_style: str) -> str:
    """Format the assistant target from a ReVA QA item."""
    # TODO(student): support "letter", "tagged", and "cot_tagged" answer styles.
    raise NotImplementedError("TODO: implement format_answer")


def iter_reva_qas(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one flat QA record at a time from nested ReVA annotations."""
    # TODO(student): iterate videos -> mcq category -> question_type -> qa list.
    raise NotImplementedError("TODO: implement iter_reva_qas")


def convert_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Convert one flat ReVA QA item into one Qwen conversation sample."""
    # TODO(student): normalize video path, optionally check video exists,
    # build question/answer, return {video, conversations, optional metadata}.
    raise NotImplementedError("TODO: implement convert_item")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reva_train/train_set.json"))
    parser.add_argument("--output", type=Path, default=Path("data/qwen_train/train.json"))
    parser.add_argument("--video-root", type=Path, default=Path("data/qwen_train"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all.")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--prompt-style", choices=["plain", "reva_eval"], default="reva_eval")
    parser.add_argument("--answer-style", choices=["letter", "tagged", "cot_tagged"], default="tagged")
    parser.add_argument("--keep-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)

    samples = []
    seen_videos = set()
    skipped_missing_video = 0
    for item in iter_reva_qas(data):
        seen_videos.add(item["video_id"])
        if args.max_videos and len(seen_videos) > args.max_videos:
            break
        sample = convert_item(item, args)
        if sample is None:
            skipped_missing_video += 1
            continue
        samples.append(sample)
        if args.max_samples and len(samples) >= args.max_samples:
            break

    dump_json(samples, args.output)
    print(f"Saved {len(samples)} Qwen training samples to {args.output}")
    if skipped_missing_video:
        print(f"Skipped {skipped_missing_video} samples with missing videos")


if __name__ == "__main__":
    main()
