#!/usr/bin/env python3
"""Check the files and commands needed for the student workflow."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA_BIN = Path(os.environ["CONDA_BIN"]) if os.environ.get("CONDA_BIN") else None
TORCHRUN_BIN = Path(os.environ["TORCHRUN_BIN"]) if os.environ.get("TORCHRUN_BIN") else None


def status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def check_file(path: Path, note: str) -> bool:
    ok = path.exists()
    print(f"[{status(ok):7}] {path.relative_to(PROJECT_ROOT)} - {note}")
    return ok


def check_dir(path: Path, note: str) -> bool:
    ok = path.is_dir()
    print(f"[{status(ok):7}] {path.relative_to(PROJECT_ROOT)} - {note}")
    return ok


def count_json_items(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and "videos" in data:
        return len(data["videos"])
    return None


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print("")

    required = [
        (PROJECT_ROOT / "data/reva_train/train_set.json", "source annotations for training-data conversion"),
        (PROJECT_ROOT / "data/qwen_train/train.json", "Qwen-format SFT data"),
        (PROJECT_ROOT / "data/reva_test/test_set.json", "ReVA evaluation questions"),
        (PROJECT_ROOT / "scripts/prepare_qwen_train_data.sh", "data conversion entry point"),
        (PROJECT_ROOT / "scripts/run_eval_qwen_base.sh", "base Qwen evaluation entry point"),
        (PROJECT_ROOT / "scripts/run_finetune_qwen.sh", "Qwen SFT/LoRA entry point"),
        (PROJECT_ROOT / "scripts/run_eval_qwen_finetuned.sh", "fine-tuned Qwen evaluation entry point"),
        (PROJECT_ROOT / "scripts/run_eval_reva_vila.sh", "VILA evaluation entry point"),
        (PROJECT_ROOT / "scripts/compare_models.sh", "metric comparison entry point"),
    ]

    all_ok = True
    for path, note in required:
        all_ok = check_file(path, note) and all_ok
    all_ok = check_dir(PROJECT_ROOT / "data/qwen_train/videos", "videos referenced by Qwen-format SFT data") and all_ok
    all_ok = check_dir(PROJECT_ROOT / "data/reva_test", "ReVA videos or extracted frame folders") and all_ok

    print("")
    for path in [
        PROJECT_ROOT / "data/reva_train/train_set.json",
        PROJECT_ROOT / "data/qwen_train/train.json",
        PROJECT_ROOT / "data/reva_test/test_set.json",
    ]:
        count = count_json_items(path)
        if count is not None:
            print(f"[INFO   ] {path.relative_to(PROJECT_ROOT)} contains {count} top-level items")

    print("")
    command_fallbacks = {
        "python3": None,
        "conda": CONDA_BIN,
        "torchrun": TORCHRUN_BIN,
    }
    for cmd, fallback in command_fallbacks.items():
        found = shutil.which(cmd) is not None or (fallback is not None and fallback.exists())
        location = shutil.which(cmd) or (str(fallback) if fallback is not None and fallback.exists() else "")
        suffix = f" ({location})" if location else ""
        print(f"[{status(found):7}] command `{cmd}`{suffix}")
        all_ok = found and all_ok

    print("")
    for name in ["MODEL_PATH", "VILA_REPO", "CONDA_ENV"]:
        value = os.environ.get(name)
        print(f"[INFO   ] {name}={value if value else '(not set)'}")

    if not all_ok:
        raise SystemExit("\nSetup check found missing required files or commands.")
    print("\nSetup check passed.")


if __name__ == "__main__":
    main()
