"""Convert ReVA_V2 test_set.json to a flat list for Qwen inference.

Student task: complete every block marked TODO(student).
"""

import json
import os
from collections import Counter
from typing import Any, Iterator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REVA_ROOT = os.environ.get("REVA_ROOT", os.path.join(PROJECT_ROOT, "data", "reva_test"))
REVA_JSON = os.environ.get("REVA_JSON", os.path.join(REVA_ROOT, "test_set.json"))
OUTPUT_JSON = os.environ.get("REVA_PREPARED_JSON", os.path.join(os.path.dirname(__file__), "reva_v2_test_set.json"))


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


def to_abs_path(file_path: str) -> str:
    """Convert an annotation video path to an absolute path under REVA_ROOT."""
    # TODO(student): remove optional dataset prefixes and join with REVA_ROOT.
    raise NotImplementedError("TODO: implement to_abs_path")


def to_rel_stem(file_path: str) -> str:
    """Convert an annotation video path to a relative stem without extension."""
    # TODO(student): normalize the same prefixes as to_abs_path and strip extension.
    raise NotImplementedError("TODO: implement to_rel_stem")


def get_qa_id(qa: dict, video_item: dict, subcategory: str, question_idx: int) -> str:
    """Return a stable QA id for evaluation."""
    # TODO(student): prefer qa_id, then global_index, otherwise synthesize from video stem.
    raise NotImplementedError("TODO: implement get_qa_id")


def iter_flat_items(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield flat Qwen-evaluation items from nested ReVA test annotations."""
    # TODO(student): iterate videos, skip missing/unreadable videos, flatten MCQs.
    raise NotImplementedError("TODO: implement iter_flat_items")


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
