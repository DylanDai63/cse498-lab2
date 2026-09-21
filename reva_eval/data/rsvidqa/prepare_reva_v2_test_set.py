"""Convert ReVA_V2 test_set.json to a flat list for Qwen inference.

Student task: complete every block marked TODO(student).
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REVA_ROOT = os.environ.get("REVA_ROOT", os.path.join(PROJECT_ROOT, "data", "reva_test"))
REVA_JSON = os.environ.get("REVA_JSON", os.path.join(REVA_ROOT, "test_set.json"))
OUTPUT_JSON = os.environ.get("REVA_PREPARED_JSON", os.path.join(os.path.dirname(__file__), "reva_v2_test_set.json"))

# Annotation paths may carry a prefix from the authors' server layout
# (e.g. "#dataset/ReVA_V2/videos/x.mp4"). Same list as scripts/check_student_setup.py
# and scripts/build_qwen_train_data.py so every tool resolves a path the same way.
DATASET_PREFIXES = ("#dataset/ReVA_V2/", "#dataset/ReVA/", "ReVA_V2/", "ReVA/")


def get_video_duration(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception as exc:
        print(f"  Warning: could not read duration for {video_path}: {exc}")
        return 0.0


def _strip_dataset_prefix(file_path: str) -> str:
    """Normalize slashes and drop a leading dataset prefix, keeping the rest relative."""
    normalized = str(file_path).replace("\\", "/")
    for prefix in DATASET_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def to_abs_path(file_path: str) -> str:
    """Convert an annotation video path to an absolute path under REVA_ROOT."""
    # TODO(student): remove optional dataset prefixes and join with REVA_ROOT.
    # REVA_ROOT is read at call time (not captured at import time) so that the unit
    # test, and the REVA_ROOT environment override, can redirect it.
    return str(Path(REVA_ROOT) / _strip_dataset_prefix(file_path))


def to_rel_stem(file_path: str) -> str:
    """Convert an annotation video path to a relative stem without extension."""
    # TODO(student): normalize the same prefixes as to_abs_path and strip extension.
    # The inference script rebuilds the file name as video_dir/<stem> + ".mp4", so the
    # stem must stay relative to REVA_ROOT and must not carry the extension.
    return os.path.splitext(_strip_dataset_prefix(file_path))[0]


def get_qa_id(qa: dict, video_item: dict, subcategory: str, question_idx: int) -> str:
    """Return a stable QA id for evaluation."""
    # TODO(student): prefer qa_id, then global_index, otherwise synthesize from video stem.
    # "Stable" means the same question gets the same id on every run and in every tool:
    # predictions are joined to ground truth by this id, and RESUME=1 relies on it.
    qa_id = qa.get("qa_id")
    if qa_id not in (None, ""):
        return str(qa_id)
    if qa.get("global_index") is not None:
        return f"REVA-G{int(qa['global_index']):06d}"
    # Last resort: video file stem + 3-letter subcategory tag + index within that list.
    # vila_eval/reva_v2.py uses the same recipe so Qwen and VILA ids stay comparable.
    stem = Path(to_rel_stem(video_item.get("file_path", ""))).name or "unknown"
    tag = re.sub(r"[^A-Za-z0-9]", "", str(subcategory))[:3].upper() or "UNK"
    return f"REVA-{stem}-{tag}-{question_idx:04d}"


def iter_flat_items(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield flat Qwen-evaluation items from nested ReVA test annotations."""
    # TODO(student): iterate videos, skip missing/unreadable videos, flatten MCQs.
    # Nesting: data["videos"][video_key]["mcq"][category][question_type] -> list of qa.
    # Output keys are exactly what reva_eval/inference_vllm_origin_number.py consumes:
    #   id, video_id (relative stem), duration, question (stem + options), answer, question_type
    skipped_missing = 0
    skipped_unreadable = 0
    for video_item in data.get("videos", {}).values():
        file_path = video_item.get("file_path")
        if not file_path:
            skipped_missing += 1
            continue

        abs_path = to_abs_path(file_path)
        if not os.path.exists(abs_path):
            skipped_missing += 1
            continue

        # Duration is needed by the inference prompt; 0 means OpenCV could not open
        # the file, so the video cannot be evaluated at all.
        duration = get_video_duration(abs_path)
        if duration <= 0:
            skipped_unreadable += 1
            continue

        video_id = to_rel_stem(file_path)
        for question_types in video_item.get("mcq", {}).values():
            if not isinstance(question_types, dict):
                continue
            for question_type, qa_list in question_types.items():
                if not isinstance(qa_list, list):
                    continue
                for question_idx, qa in enumerate(qa_list):
                    options = qa.get("options", {})
                    # Sorted "A. text" lines: same option format as the training data.
                    option_lines = [f"{label}. {options[label]}" for label in sorted(options)]
                    yield {
                        "id": get_qa_id(qa, video_item, question_type, question_idx),
                        "video_id": video_id,
                        "duration": round(duration, 2),
                        "question": "\n".join([str(qa.get("question", ""))] + option_lines),
                        "answer": str(qa.get("correct_answer", "")).strip().upper(),
                        "question_type": question_type,
                    }

    # Skipped videos shrink the evaluated question set, so they are reported loudly
    # instead of disappearing: Qwen and VILA must be compared on the same questions.
    if skipped_missing or skipped_unreadable:
        print(
            f"  Warning: skipped {skipped_missing} missing and {skipped_unreadable} unreadable videos",
            file=sys.stderr,
        )


def main():
    with open(REVA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    subcats = Counter()
    for item in iter_flat_items(data):
        items.append(item)
        subcats[item.get("question_type", "unknown")] += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(items)} QA items to {OUTPUT_JSON}")
    print("\nSubcategory distribution:")
    for subcat, count in sorted(subcats.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {subcat}: {count}")


if __name__ == "__main__":
    main()
