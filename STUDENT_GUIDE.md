# Student Guide: ReVA Model Evaluation and Qwen Fine-Tuning

This project gives you one complete visual-language-model experiment loop:

1. Convert ReVA annotations into Qwen training data.
2. Evaluate a base Qwen-VL model on ReVA.
3. Fine-tune Qwen-VL with LoRA/SFT.
4. Evaluate the fine-tuned Qwen checkpoint on the same ReVA split.
5. Evaluate VILA as a second baseline.
6. Compare model metrics and inspect errors.

Run all commands from the project root:

```bash
cd ~/code/CSE_Homeworks/qwen_reva_assignment
```

If your directory is different, use that path instead.

## Official Resources

- ReVA dataset: [ReVA-Benchmark/ReVA](https://huggingface.co/datasets/ReVA-Benchmark/ReVA)
- Qwen base model: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

Your instructor may have already downloaded both resources to the teaching server. Otherwise, use:

```bash
export REVA_DATA_ROOT=/path/to/ReVA
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct

hf download ReVA-Benchmark/ReVA --repo-type dataset --local-dir "$REVA_DATA_ROOT"
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$QWEN3_VL_MODEL_PATH"
```

The full ReVA download is about 29.9 GB. Do not flatten or rename its video directories: paths in the annotation files are relative to the ReVA root.

## 0. Check Setup

```bash
python3 scripts/check_student_setup.py
```

The check should find these required files:

```text
data/reva_train/train_set.json
data/qwen_train/train.json
data/reva_test/test_set.json
data/qwen_train/videos/
```

If `data/reva_test/test_set.json` is missing, copy or symlink the instructor-provided ReVA test subset into `data/reva_test/`.

## 1. Prepare Qwen Training Data

```bash
bash scripts/prepare_qwen_train_data.sh
```

Useful smaller or larger runs:

```bash
MAX_SAMPLES=50 bash scripts/prepare_qwen_train_data.sh
MAX_SAMPLES=1000 bash scripts/prepare_qwen_train_data.sh
MAX_SAMPLES=0 bash scripts/prepare_qwen_train_data.sh
```

The output is:

```text
data/qwen_train/train.json
```

The preparation script checks video references by default and preserves the existing output if no referenced video can be found. A zero-sample result therefore indicates an incorrect dataset path, not a usable training set.

Each sample has a video path plus a two-turn conversation:

```json
{
  "video": "videos/example.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion..."},
    {"from": "gpt", "value": "<answer>B</answer>"}
  ]
}
```

## 2. Evaluate Base Qwen

Set the base Qwen checkpoint path for your server:

```bash
export MODEL_PATH=<QWEN3_VL_MODEL_PATH>
```

Then run:

```bash
bash scripts/run_eval_qwen_base.sh
```

The main outputs are:

```text
outputs/qwen_base/qwen_base/result.csv
outputs/qwen_base/qwen_base/result.json
```

`result.csv` contains completion, total accuracy, and subcategory accuracies. Missing or unparsable predictions count as incorrect instead of disappearing from the denominator.

## 3. Fine-Tune Qwen

```bash
MODEL_PATH=<QWEN3_VL_MODEL_PATH> \
NPROC_PER_NODE=1 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=4 \
EPOCHS=1 \
bash scripts/run_finetune_qwen.sh
```

Default output:

```text
outputs/qwen_reva_sft/
```

The training script uses the dataset registered as `reva_train_small` in:

```text
qwen_finetune/qwenvl/data/__init__.py
```

For a short debugging run, reduce the training data first:

```bash
MAX_SAMPLES=20 bash scripts/prepare_qwen_train_data.sh
EPOCHS=1 SAVE_STEPS=20 bash scripts/run_finetune_qwen.sh
```

## 4. Evaluate Fine-Tuned Qwen

Point `MODEL_PATH` to the checkpoint produced by fine-tuning:

```bash
MODEL_PATH=outputs/qwen_reva_sft/checkpoint-2 \
bash scripts/run_eval_qwen_finetuned.sh
```

`checkpoint-2` is the expected smoke-test example for the bundled two-sample training split. If your run saves a different checkpoint number, replace it with the actual checkpoint directory.

The main outputs are:

```text
outputs/qwen_sft/qwen_sft/result.csv
outputs/qwen_sft/qwen_sft/result.json
```

## 5. Evaluate VILA Baseline

Point `VILA_REPO` to a working VILA checkout on the server:

```bash
VILA_REPO=<VILA_REPO> \
MODEL_PATH=Efficient-Large-Model/VILA1.5-3b \
bash scripts/run_eval_reva_vila.sh
```

For a smoke test:

```bash
MAX_QUESTIONS=10 \
VILA_REPO=<VILA_REPO> \
MODEL_PATH=Efficient-Large-Model/VILA1.5-3b \
bash scripts/run_eval_reva_vila.sh
```

The main outputs are:

```text
outputs/vila_reva_v2/metrics.json
outputs/vila_reva_v2/outputs.jsonl
```

## 6. Compare Models

```bash
bash scripts/compare_models.sh
```

This writes:

```text
outputs/model_comparison.csv
```

The table compares:

- Qwen base
- Qwen fine-tuned
- VILA base

Qwen and VILA must be compared on the same question set. Check both `num_questions` and `num_answered`; do not report accuracy alone.

For the exact runnable server commands on `<REMOTE_HOST>`, see:

```text
INSTRUCTIONS.md
```

## 7. What to Submit

Submit:

- the commands you ran,
- `outputs/model_comparison.csv`,
- the base Qwen `result.csv`,
- the fine-tuned Qwen `result.csv`,
- the VILA `metrics.json`,
- three to five qualitative examples from `result.json` or `outputs.jsonl`.

For each qualitative example, label the likely issue as visual perception, temporal reasoning, option parsing, or overfitting.
