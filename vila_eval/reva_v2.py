import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime
from typing import Any

# Annotation paths may carry a prefix from the authors' server layout. Same list as
# scripts/check_student_setup.py so every tool in this repo resolves paths the same way.
DATASET_PREFIXES = ("#dataset/ReVA_V2/", "#dataset/ReVA/", "ReVA_V2/", "ReVA/")

_FLAGS = re.DOTALL | re.IGNORECASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timestamp-output", action="store_true")
    return parser.parse_args()


def _video_stem(file_path: str) -> str:
    """File name without directories or extension: "#dataset/ReVA_V2/videos/demo.mp4" -> "demo"."""
    normalized = str(file_path).replace("\\", "/")
    return os.path.splitext(normalized.rsplit("/", 1)[-1])[0] or "unknown"


def load_instances(question_file: str) -> list[dict[str, Any]]:
    """Flatten nested ReVA annotations into VILA evaluation instances."""
    # TODO(student): read question_file, flatten instances, and assign stable IDs
    # using qa_id, then global_index, then video/subcategory/question index.
    # Nesting: data["videos"][video_id]["mcq"][category][subcategory] -> list of qa.
    # The id recipe is identical to reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py,
    # so a Qwen prediction and a VILA prediction for the same question share one id.
    with open(question_file, encoding="utf-8") as f:
        data = json.load(f)

    instances: list[dict[str, Any]] = []
    for video_id, video_item in data.get("videos", {}).items():
        file_path = video_item.get("file_path", "")
        for category, subcategories in video_item.get("mcq", {}).items():
            if not isinstance(subcategories, dict):
                continue
            for subcategory, qa_list in subcategories.items():
                if not isinstance(qa_list, list):
                    continue
                for question_idx, qa in enumerate(qa_list):
                    if qa.get("qa_id") not in (None, ""):
                        qa_id = str(qa["qa_id"])
                    elif qa.get("global_index") is not None:
                        qa_id = f"REVA-G{int(qa['global_index']):06d}"
                    else:
                        tag = re.sub(r"[^A-Za-z0-9]", "", str(subcategory))[:3].upper() or "UNK"
                        qa_id = f"REVA-{_video_stem(file_path)}-{tag}-{question_idx:04d}"

                    instances.append({
                        "qa_id": qa_id,
                        "video_id": video_id,
                        "video_path": file_path,
                        "dataset_name": video_item.get("dataset_name"),
                        "category": category,
                        "subcategory": subcategory,
                        "question": qa.get("question", ""),
                        "options": qa.get("options", {}),
                        # Upper-cased because main() compares it with the parsed letter.
                        "correct_answer": str(qa.get("correct_answer", "")).strip().upper(),
                        "reasoning": qa.get("reasoning", ""),
                        "example": qa.get("example", ""),
                    })
    return instances


def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    """Resolve a ReVA annotation path to an existing local video path."""
    # TODO(student): try raw_path, dataset_root/raw_path, and paths with dataset_prefix removed.
    normalized = str(raw_path).replace("\\", "/")
    root = Path(dataset_root)

    # Candidates, most literal first. With prefix "#dataset" the annotation
    # "#dataset/ReVA_V2/videos/x.mp4" becomes "ReVA_V2/videos/x.mp4"; our download puts
    # videos directly under the ReVA root, so the dataset folder name is dropped as well.
    relative_forms = [normalized]
    if dataset_prefix and normalized.startswith(dataset_prefix):
        without_prefix = normalized[len(dataset_prefix):].lstrip("/")
        relative_forms.append(without_prefix)
        if "/" in without_prefix:
            relative_forms.append(without_prefix.split("/", 1)[1])
    for prefix in DATASET_PREFIXES:
        if normalized.startswith(prefix):
            relative_forms.append(normalized[len(prefix):])

    candidates = [Path(normalized)] + [root / form for form in relative_forms]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fail loudly: silently skipping a video would shrink the evaluated question set.
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve video path {raw_path!r}; tried: {tried}")


def build_prompt(question: str, options: dict[str, str]) -> str:
    """Build the VILA multiple-choice prompt."""
    # TODO(student): include question, sorted options, and one-letter instruction.
    # Options are sorted so the model always sees A -> D regardless of JSON key order.
    lines = [question]
    lines.extend(f"{label}. {options[label]}" for label in sorted(options))
    lines.append("Answer with only the option letter from the given choices.")
    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str]) -> str | None:
    """Parse a choice letter from VILA raw response."""
    # TODO(student): robustly parse A/B/C/D from raw output or matching option text.
    # Returns None when the output is unclear: an unparsed answer is scored as incorrect
    # (see summarize), which is honest; guessing a letter would distort accuracy.
    # The letter tiers mirror scripts/score_reva_predictions.py. The logic is repeated here
    # because this file runs inside the separate VILA environment and cannot import it.
    valid = {str(label).strip().upper() for label in options}
    text = "" if response is None else str(response)

    tagged = [inner.strip() for inner in re.findall(r"<answer>(.*?)</answer>", text, flags=_FLAGS) if inner.strip()]
    if tagged:
        text = tagged[-1]
    else:
        text = re.sub(r"<think>.*?</think>", " ", text, flags=_FLAGS)
        text = re.sub(r"<think>.*", " ", text, flags=_FLAGS)
    text = text.strip()
    if not text:
        return None

    def accept(letter: str) -> str | None:
        letter = letter.upper()
        return letter if letter in valid else None

    # Tier 1 - the whole reply is one letter, maybe wrapped: "B", "(b)", "B.".
    match = re.fullmatch(r"[\s\(\[\"'*]*([A-Za-z])[\s\)\]\.\:\"'*]*", text)
    if match and accept(match.group(1)):
        return accept(match.group(1))

    # Tier 2 - the reply starts with a letter label: "B. video", "(b) video".
    match = re.match(r"[\s\(\[*]*([A-Z])\s*[\)\]\.\:,\-]", text) or re.match(r"\s*\(?([a-z])\)", text)
    if match and accept(match.group(1)):
        return accept(match.group(1))

    # Tier 3 - an explicit cue: "The answer is B.". Upper-case only, so that
    # "the answer is a video" is not read as option A.
    cues = re.findall(
        r"(?i:\banswer\b)(?i:\s+(?:is|would\s+be|should\s+be))?\s*[:\-=]?\s*\*{0,2}\(?([A-Z])\b",
        text,
    )
    cues = [letter for letter in cues if letter in valid]
    if cues:
        return cues[-1]

    # Tier 4 - option text instead of a letter: "video" or "It is a video.".
    # Accepted only when exactly one option matches; an option contained in a longer
    # matching option ("car" inside "red car") does not count as a second match.
    lowered = text.lower()
    matched = []
    for label, option_text in options.items():
        needle = str(option_text).strip().lower()
        if needle and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", lowered):
            matched.append((str(label).strip().upper(), needle))
    matched = [
        (label, needle)
        for label, needle in matched
        if not any(needle != other and needle in other for _, other in matched)
    ]
    if len(matched) == 1:
        return matched[0][0]

    # Tier 5 - exactly one distinct stand-alone capital option letter in the reply.
    # "A" followed by a lower-case word is the English article, not an option.
    without_article = re.sub(r"\bA(?=\s+[a-z])", " ", text)
    letters = {letter for letter in re.findall(r"\b([A-Z])\b", without_article) if letter in valid}
    if len(letters) == 1:
        return letters.pop()
    return None


def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _group_stats(records: list[dict]) -> dict:
    """Counts for one group. Accuracy divides by ALL questions, answered or not."""
    num_questions = len(records)
    num_answered = sum(1 for record in records if record.get("pred_letter") not in (None, ""))
    num_correct = sum(1 for record in records if record.get("is_correct"))
    return {
        "num_questions": num_questions,
        "num_answered": num_answered,
        "num_correct": num_correct,
        "accuracy": num_correct / num_questions if num_questions else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    """Compute total, category, and subcategory accuracy for VILA outputs."""
    # TODO(student): compute num_questions, num_answered, accuracy, by_category, by_subcategory.
    # num_answered is reported next to accuracy so a model cannot look good by answering
    # only the questions it finds easy: unanswered questions stay in the denominator.
    # scripts/compare_model_metrics.py reads accuracy, num_questions and num_answered.
    by_category: dict[str, list[dict]] = {}
    by_subcategory: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(str(record.get("category", "unknown")), []).append(record)
        by_subcategory.setdefault(str(record.get("subcategory", "unknown")), []).append(record)

    metrics = _group_stats(records)
    metrics["by_category"] = {name: _group_stats(group) for name, group in sorted(by_category.items())}
    metrics["by_subcategory"] = {name: _group_stats(group) for name, group in sorted(by_subcategory.items())}
    return metrics


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir if args.resume or not args.timestamp_output else base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"
    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content([Video(video_path), prompt], generation_config=generation_config)
        pred_letter = parse_choice(response, instance["options"])

        outputs.append({
            "qa_id": instance["qa_id"],
            "video_id": instance["video_id"],
            "video_path": video_path,
            "dataset_name": instance.get("dataset_name"),
            "category": instance["category"],
            "subcategory": instance["subcategory"],
            "question": instance["question"],
            "options": instance["options"],
            "prompt": prompt,
            "raw_response": response,
            "pred_letter": pred_letter,
            "correct_answer": instance["correct_answer"],
            "is_correct": pred_letter == instance["correct_answer"],
            "reasoning": instance.get("reasoning", ""),
            "example": instance.get("example", ""),
        })
        save_jsonl(str(outputs_path), outputs)

    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
