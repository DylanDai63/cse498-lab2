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
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(video_root).as_posix()
    except ValueError:
        return path.as_posix()


# ReVA annotations were written on the authors' server, where video paths carry a
# dataset prefix such as "#dataset/ReVA_V2/Hawk_UAV/...". Our copy lives directly
# under the ReVA root, so the prefix must go. Same list as scripts/check_student_setup.py
# so that the converter and the setup checker always agree on what a path means.
DATASET_PREFIXES = ("#dataset/ReVA_V2/", "#dataset/ReVA/", "ReVA_V2/", "ReVA/")


def strip_dataset_prefix(file_path: str) -> str:
    """Drop a leading dataset prefix and normalize slashes, keeping the rest relative."""
    normalized = file_path.replace("\\", "/")
    for prefix in DATASET_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def format_options(options: dict[str, str]) -> str:
    """Return sorted multiple-choice options, one option per line."""
    # TODO(student): sort option labels and format each line as "A. option text".
    # Labels are sorted because the annotation JSON does not guarantee A->D order;
    # the model must always see options in the same order it will see them at eval time.
    # "A. text" mirrors the option format used by reva_eval's inference prompt.
    lines = []
    for label in sorted(options):
        lines.append(f"{label}. {options[label]}")
    return "\n".join(lines)


def format_question(question: str, options: dict[str, str], prompt_style: str) -> str:
    """Build the human message used by Qwen SFT."""
    # TODO(student): call format_options and support "plain" and "reva_eval" styles.
    # "<video>" is the placeholder the Qwen-VL data loader replaces with video tokens.
    # "reva_eval" copies the wording of MY_PROMPT_TEMPLATE in reva_eval/eval_reva_v2.sh
    # (minus the duration line) so that training prompts match evaluation prompts.
    options_text = format_options(options)
    if prompt_style == "plain":
        return f"<video>\nQuestion: {question}\n{options_text}"
    if prompt_style == "reva_eval":
        return (
            "<video>\n"
            "Please carefully watch the video and answer the multiple-choice question below.\n"
            "Think step-by-step within <think> </think> tags, then provide only the letter "
            "of the correct option within <answer> </answer> tags.\n"
            f"Question: {question}\n{options_text}"
        )
    raise ValueError(f"unknown prompt_style: {prompt_style}")


def format_answer(qa: dict[str, Any], answer_style: str) -> str:
    """Format the assistant target from a ReVA QA item."""
    # TODO(student): support "letter", "tagged", and "cot_tagged" answer styles.
    # The letter is upper-cased because some annotations store "b" while the options
    # dict is keyed "B"; the scorer compares upper-case letters.
    letter = str(qa["correct_answer"]).strip().upper()
    if answer_style == "letter":
        return f"Answer: {letter}"
    if answer_style == "tagged":
        # Default for the student run: train only the final letter inside <answer> tags,
        # avoiding noisy free-text reasoning from the annotation file.
        return f"<answer>{letter}</answer>"
    if answer_style == "cot_tagged":
        # Teacher-experiment style: also train on the annotation's rationale.
        # Fall back to the tagged form when a QA item has no reasoning text.
        reasoning = str(qa.get("reasoning", "")).strip()
        if not reasoning:
            return f"<answer>{letter}</answer>"
        return f"<think>{reasoning}</think><answer>{letter}</answer>"
    raise ValueError(f"unknown answer_style: {answer_style}")


def iter_reva_qas(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one flat QA record at a time from nested ReVA annotations."""
    # TODO(student): iterate videos -> mcq category -> question_type -> qa list.
    # ReVA nests QA items four levels deep:
    #   data["videos"][video_id]["mcq"][category][question_type] -> list of qa dicts
    # Each yielded record carries the video path plus the category/question_type labels,
    # so downstream code never has to walk the nested structure again.
    for video_id, video_item in data.get("videos", {}).items():
        file_path = video_item.get("file_path", "")
        mcq = video_item.get("mcq", {})
        for category, question_types in mcq.items():
            if not isinstance(question_types, dict):
                continue
            for question_type, qa_list in question_types.items():
                if not isinstance(qa_list, list):
                    continue
                for qa in qa_list:
                    record = dict(qa)
                    record["video_id"] = video_id
                    record["file_path"] = file_path
                    record["dataset_name"] = video_item.get("dataset_name", "")
                    record["category"] = category
                    record["question_type"] = question_type
                    yield record


def convert_item(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Convert one flat ReVA QA item into one Qwen conversation sample."""
    # TODO(student): normalize video path, optionally check video exists,
    # build question/answer, return {video, conversations, optional metadata}.
    # The "video" field must be relative to args.video_root, because
    # qwen_finetune/qwenvl/data/__init__.py registers data/qwen_train as data_path and
    # the loader joins data_path / video.
    video = strip_dataset_prefix(normalize_video_path(item["file_path"], args.video_root))
    if not video:
        return None

    # Returning None (instead of raising) lets main() count skipped samples and refuse
    # to overwrite train.json when nothing resolves - a zero-sample result should mean
    # "wrong dataset path", not "empty training set".
    if getattr(args, "require_video", False):
        candidate = Path(video)
        if not candidate.is_absolute():
            candidate = Path(args.video_root) / candidate
        if not candidate.exists():
            return None

    human_message = format_question(item["question"], item["options"], args.prompt_style)
    assistant_message = format_answer(item, args.answer_style)

    sample: dict[str, Any] = {
        "video": video,
        "conversations": [
            {"from": "human", "value": human_message},
            {"from": "gpt", "value": assistant_message},
        ],
    }

    # Metadata is not consumed by training; it is kept so a sample in train.json can be
    # traced back to its ReVA qa_id when analysing results.
    if getattr(args, "keep_metadata", False):
        sample["metadata"] = {
            "qa_id": item.get("qa_id"),
            "video_id": item.get("video_id"),
            "dataset_name": item.get("dataset_name"),
            "category": item.get("category"),
            "question_type": item.get("question_type"),
            "correct_answer": str(item.get("correct_answer", "")).strip().upper(),
        }
    return sample


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

    if args.require_video and not samples:
        raise SystemExit(
            "No training samples reference an existing video. "
            "Check QWEN_VIDEO_ROOT and the downloaded ReVA directory; output was not modified."
        )

    dump_json(samples, args.output)
    print(f"Saved {len(samples)} Qwen training samples to {args.output}")
    if skipped_missing_video:
        print(f"Skipped {skipped_missing_video} samples with missing videos")


if __name__ == "__main__":
    main()
