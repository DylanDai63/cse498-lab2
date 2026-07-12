from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_qwen_train_data_helpers():
    mod = load_module("build_qwen_train_data", "scripts/build_qwen_train_data.py")
    qa = {
        "question": "What is shown?",
        "options": {"B": "a video clip", "A": "a still image"},
        "correct_answer": "b",
        "reasoning": "The input is a video.",
    }

    assert mod.format_options(qa["options"]) == "A. a still image\nB. a video clip"

    plain = mod.format_question(qa["question"], qa["options"], "plain")
    assert plain.startswith("<video>\nQuestion: What is shown?")
    assert "A. a still image" in plain and "B. a video clip" in plain

    reva_eval = mod.format_question(qa["question"], qa["options"], "reva_eval")
    assert reva_eval.startswith("<video>\nPlease carefully watch the video")
    assert "<answer> </answer>" in reva_eval

    assert mod.format_answer(qa, "letter") == "Answer: B"
    assert mod.format_answer(qa, "tagged") == "<answer>B</answer>"
    assert mod.format_answer(qa, "cot_tagged") == "<think>The input is a video.</think><answer>B</answer>"


def test_build_qwen_train_data_conversion(tmp_path):
    mod = load_module("build_qwen_train_data", "scripts/build_qwen_train_data.py")
    video_root = tmp_path / "qwen_train"
    (video_root / "videos").mkdir(parents=True)
    (video_root / "videos/demo.mp4").write_bytes(b"fake")

    data = {
        "videos": {
            "vid1": {
                "file_path": "videos/demo.mp4",
                "mcq": {
                    "video_qa": {
                        "basic": [
                            {
                                "qa_id": "Q1",
                                "question": "What is shown?",
                                "options": {"A": "image", "B": "video"},
                                "correct_answer": "B",
                            }
                        ]
                    }
                },
            }
        }
    }

    items = list(mod.iter_reva_qas(data))
    assert len(items) == 1
    assert items[0]["video_id"] == "vid1"
    assert items[0]["category"] == "video_qa"
    assert items[0]["question_type"] == "basic"

    class Args:
        pass

    args = Args()
    args.video_root = video_root
    args.require_video = True
    args.prompt_style = "reva_eval"
    args.answer_style = "tagged"
    args.keep_metadata = True

    sample = mod.convert_item(items[0], args)
    assert sample is not None
    assert sample["video"] == "videos/demo.mp4"
    assert sample["conversations"][0]["from"] == "human"
    assert sample["conversations"][1] == {"from": "gpt", "value": "<answer>B</answer>"}
    assert sample["metadata"]["qa_id"] == "Q1"


def test_prepare_reva_v2_flatten(monkeypatch, tmp_path):
    mod = load_module("prepare_reva", "reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py")
    monkeypatch.setattr(mod, "REVA_ROOT", str(tmp_path))
    monkeypatch.setattr(mod, "get_video_duration", lambda path: 3.5)
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/demo.mp4").write_bytes(b"fake")

    assert mod.to_abs_path("#dataset/ReVA_V2/videos/demo.mp4") == str(tmp_path / "videos/demo.mp4")
    assert mod.to_rel_stem("#dataset/ReVA_V2/videos/demo.mp4") == "videos/demo"

    qa = {"qa_id": "Q1", "global_index": 7}
    assert mod.get_qa_id(qa, {"file_path": "videos/demo.mp4"}, "basic", 0) == "Q1"
    assert mod.get_qa_id({"global_index": 7}, {"file_path": "videos/demo.mp4"}, "basic", 0) == "REVA-G000007"

    data = {
        "videos": {
            "vid1": {
                "file_path": "videos/demo.mp4",
                "mcq": {
                    "video_qa": {
                        "basic": [
                            {
                                "qa_id": "Q1",
                                "question": "What?",
                                "options": {"A": "image", "B": "video"},
                                "correct_answer": "B",
                            }
                        ]
                    }
                },
            }
        }
    }
    items = list(mod.iter_flat_items(data))
    assert items == [
        {
            "id": "Q1",
            "video_id": "videos/demo",
            "duration": 3.5,
            "question": "What?\nA. image\nB. video",
            "answer": "B",
            "question_type": "basic",
        }
    ]


def test_score_reva_predictions(tmp_path):
    mod = load_module("score_reva", "scripts/score_reva_predictions.py")
    assert mod.extract_answer("<think>x</think><answer> b </answer>") == "b"
    assert mod.extract_answer("Answer: C") == "Answer: C"
    assert mod.extract_letter("<answer>b</answer>") == "B"
    assert mod.extract_letter("The answer is C because...") == "C"

    out_dir = tmp_path / "preds"
    out_dir.mkdir()
    pred_path = out_dir / "pred.json"
    pred_path.write_text(
        json.dumps({"id": "Q1", "pred": "<answer>B</answer>", "question_type": "basic"}) + "\n"
        + json.dumps({"id": "Q2", "pred": "A", "question_type": "basic"}) + "\n",
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text("{}\n", encoding="utf-8")

    preds = mod.load_predictions(out_dir)
    assert set(preds) == {"Q1", "Q2"}

    gt_list = [
        {"id": "Q1", "answer": "B", "question_type": "basic"},
        {"id": "Q2", "answer": "C", "question_type": "basic"},
    ]
    results, csv_text = mod.score_predictions(preds, gt_list)
    assert results["Q1"]["acc"] == 1
    assert results["Q2"]["acc"] == 0
    assert "Total: 1/2 = 50.00%" in csv_text
    assert "basic" in csv_text


def test_vila_helpers(tmp_path):
    mod = load_module("vila_reva", "vila_eval/reva_v2.py")
    question_file = tmp_path / "test_set.json"
    question_file.write_text(
        json.dumps(
            {
                "videos": {
                    "vid1": {
                        "file_path": "#dataset/ReVA_V2/videos/demo.mp4",
                        "dataset_name": "demo",
                        "mcq": {
                            "video_qa": {
                                "basic": [
                                    {
                                        "qa_id": "Q1",
                                        "question": "What?",
                                        "options": {"A": "image", "B": "video"},
                                        "correct_answer": "B",
                                        "reasoning": "Because it is a video.",
                                    }
                                ]
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    instances = mod.load_instances(str(question_file))
    assert len(instances) == 1
    assert instances[0]["qa_id"] == "Q1"
    assert instances[0]["subcategory"] == "basic"

    root = tmp_path / "root"
    (root / "videos").mkdir(parents=True)
    video_path = root / "videos/demo.mp4"
    video_path.write_bytes(b"fake")
    assert mod.resolve_video_path("#dataset/ReVA_V2/videos/demo.mp4", str(root), "#dataset") == str(video_path)

    prompt = mod.build_prompt("What?", {"B": "video", "A": "image"})
    assert prompt.splitlines() == ["What?", "A. image", "B. video", "Answer with only the option letter from the given choices."]

    assert mod.parse_choice("The answer is B.", {"A": "image", "B": "video"}) == "B"
    assert mod.parse_choice("video", {"A": "image", "B": "video"}) == "B"
    assert mod.parse_choice("unknown", {"A": "image", "B": "video"}) is None

    metrics = mod.summarize(
        [
            {"pred_letter": "B", "is_correct": True, "category": "video_qa", "subcategory": "basic"},
            {"pred_letter": None, "is_correct": False, "category": "video_qa", "subcategory": "basic"},
        ]
    )
    assert metrics["num_questions"] == 2
    assert metrics["num_answered"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["by_subcategory"]["basic"]["accuracy"] == 0.5
