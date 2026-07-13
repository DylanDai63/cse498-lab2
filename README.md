# Qwen + ReVA Assignment Project

## Student Assignment

Start from `ASSIGNMENT.md`. The files marked with `TODO(student)` are intentionally incomplete. The teacher reference implementation is kept separately at `<PROJECT_ROOT>`.


This project packages the useful parts of the original two-server workflow into one teaching-oriented project:

- `qwen_finetune/`: Qwen-VL SFT/LoRA training code included for this assignment.
- `reva_eval/`: ReVA evaluation code copied from the frost snapshot.
- `vila_eval/`: VILA ReVA evaluation adapter included for this assignment.
- `data/reva_train/`: ReVA training annotations.
- `data/qwen_train/`: small Qwen-format training data.
- `data/reva_test/`: expected location for ReVA `test_set.json` and videos.
- `outputs/`: default location for checkpoints and evaluation outputs.

For the student-facing workflow, start with:

```text
STUDENT_GUIDE.md
INSTRUCTIONS.md
```

The guides walk through setup checks, base-model evaluation, Qwen fine-tuning, fine-tuned evaluation, VILA baseline evaluation, and metric comparison.

## 1. Prepare Data

Start from ReVA training annotations:

```text
data/reva_train/train_set.json
```

Convert ReVA annotations into Qwen/LLaVA conversation format:

```bash
bash scripts/prepare_qwen_train_data.sh
```

By default this creates a 200-sample subset at:

```text
data/qwen_train/train.json
```

Use more or fewer samples with:

```bash
MAX_SAMPLES=1000 bash scripts/prepare_qwen_train_data.sh
MAX_SAMPLES=0 bash scripts/prepare_qwen_train_data.sh
```

The generated training data uses this format:

```json
{
  "video": "videos/example.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion..."},
    {"from": "gpt", "value": "<answer>B</answer>"}
  ]
}
```

The converter also supports chain-of-thought style targets for teacher experiments:

```bash
ANSWER_STYLE=cot_tagged bash scripts/prepare_qwen_train_data.sh
```

For the default student-ready run, `<answer>B</answer>` is safer because it avoids training on noisy or inconsistent reasoning text in the annotation file.

ReVA evaluation expects:

```text
data/reva_test/test_set.json
data/reva_test/<video files or frame folders>
```

The original server path was `<REVA_DATA_ROOT>`; copy or symlink the needed ReVA test subset into `data/reva_test/`.

## 2. Fine-Tune Qwen

Default command:

```bash
bash scripts/run_finetune_qwen.sh
```

Useful overrides:

```bash
MODEL_PATH=<QWEN3_VL_MODEL_PATH> \
NPROC_PER_NODE=1 \
BATCH_SIZE=1 \
EPOCHS=1 \
SAVE_STEPS=1000 \
bash scripts/run_finetune_qwen.sh
```

The dataset name defaults to `reva_train_small`, registered in:

```text
qwen_finetune/qwenvl/data/__init__.py
```

The default output directory is:

```text
outputs/qwen_reva_sft
```

## 3. Evaluate on ReVA

Evaluate the base model:

```bash
MODEL_PATH=<QWEN3_VL_MODEL_PATH> \
bash scripts/run_eval_qwen_base.sh
```

Evaluate a fine-tuned checkpoint:

```bash
MODEL_PATH=/path/to/checkpoint \
bash scripts/run_eval_qwen_finetuned.sh
```

Useful overrides:

```bash
REVA_ROOT=/path/to/ReVA_V2 \
REVA_JSON=/path/to/ReVA_V2/test_set.json \
NUM_CHUNKS=1 \
MAX_FRAMES=32 \
CONDA_ENV=verl \
bash scripts/run_eval_reva.sh
```

Evaluation writes:

```text
outputs/qwen_base/qwen_base/result.json
outputs/qwen_base/qwen_base/result.csv
outputs/qwen_sft/qwen_sft/result.json
outputs/qwen_sft/qwen_sft/result.csv
```

## 4. Evaluate VILA on ReVA

VILA is included as a second VLM for model comparison. The intended teaching use is evaluation, not VILA fine-tuning.

On the server, point `VILA_REPO` to the installed VILA repository:

```bash
VILA_REPO=<VILA_REPO> \
MODEL_PATH=Efficient-Large-Model/VILA1.5-3b \
CONDA_ENV=vila \
bash scripts/run_eval_reva_vila.sh
```

Useful smoke-test option:

```bash
MAX_QUESTIONS=10 bash scripts/run_eval_reva_vila.sh
```

VILA evaluation writes:

```text
outputs/vila_reva_v2/outputs.jsonl
outputs/vila_reva_v2/metrics.json
```

After running base Qwen, fine-tuned Qwen, and VILA evaluation, compare all three:

```bash
bash scripts/compare_models.sh
```

This writes:

```text
outputs/model_comparison.csv
```

## 5. Assignment Scope

For a student assignment, keep the required workflow to:

1. Convert ReVA annotations into Qwen training data.
2. Run Qwen baseline on ReVA.
3. Run VILA baseline on ReVA.
4. Fine-tune Qwen on a small Qwen-format training set.
5. Run ReVA evaluation on the fine-tuned Qwen checkpoint.
6. Compare Qwen baseline, VILA baseline, and fine-tuned Qwen, then analyze several success and failure cases.
