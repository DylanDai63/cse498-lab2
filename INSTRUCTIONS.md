# Runnable Instructions for Students

Run everything from:

```bash
cd <PROJECT_ROOT>
export PATH=<CONDA_ROOT>/bin:$PATH
export MODEL_PATH=<QWEN3_VL_MODEL_PATH>
```

## 1. Prepare Demo Data

The bundled project contains one demo video. This command creates a tiny ReVA-style train/test split so the full workflow can run before the full ReVA videos are available.

```bash
bash scripts/setup_demo_data.sh
python3 scripts/check_student_setup.py
```

## 2. Qwen Baseline Evaluation

```bash
CONDA_ENV=qwen2 \
MODEL_PATH=<QWEN3_VL_MODEL_PATH> \
BACKEND=transformers \
MAX_FRAMES=4 \
bash scripts/run_eval_qwen_base.sh
```

Outputs:

```text
outputs/qwen_base/qwen_base/result.csv
outputs/qwen_base/qwen_base/result.json
```

## 3. Qwen Fine-Tuning

Use this smoke-test setting first:

```bash
CONDA_ENV=qwen2 \
MODEL_PATH=<QWEN3_VL_MODEL_PATH> \
NPROC_PER_NODE=1 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=1 \
EPOCHS=1 \
SAVE_STEPS=1 \
MAX_PIXELS=50176 \
VIDEO_MAX_FRAMES=4 \
VIDEO_FPS=1 \
MODEL_MAX_LENGTH=2048 \
USE_DEEPSPEED=0 \
bash scripts/run_finetune_qwen.sh
```

The output checkpoint is written under:

```text
outputs/qwen_reva_sft/
```

For a larger run after the smoke test, use two GPUs and more accumulation:

```bash
CONDA_ENV=qwen2 \
MODEL_PATH=<QWEN3_VL_MODEL_PATH> \
NPROC_PER_NODE=2 \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=4 \
EPOCHS=1 \
SAVE_STEPS=50 \
bash scripts/run_finetune_qwen.sh
```

## 4. Fine-Tuned Qwen Evaluation

Replace `checkpoint-2` with the actual checkpoint directory produced by training if your run saves a different checkpoint number.

```bash
CONDA_ENV=qwen2 \
MODEL_PATH=outputs/qwen_reva_sft/checkpoint-2 \
MODEL_BASE=<QWEN3_VL_MODEL_PATH> \
BACKEND=transformers \
MAX_FRAMES=4 \
bash scripts/run_eval_qwen_finetuned.sh
```

Outputs:

```text
outputs/qwen_sft/qwen_sft/result.csv
outputs/qwen_sft/qwen_sft/result.json
```

## 5. VILA Baseline Evaluation

```bash
CONDA_ENV=vila \
VILA_REPO=<VILA_REPO> \
MODEL_PATH=Efficient-Large-Model/VILA1.5-3b \
MAX_QUESTIONS=2 \
NUM_VIDEO_FRAMES=4 \
bash scripts/run_eval_reva_vila.sh
```

Outputs:

```text
outputs/vila_reva_v2/metrics.json
outputs/vila_reva_v2/outputs.jsonl
```

## 6. Compare Results

```bash
python3 scripts/compare_model_metrics.py
cat outputs/model_comparison.csv
```

## 7. Full ReVA Data

The demo flow is only for checking that code, environments, and commands work. For the real assignment, replace:

```text
data/reva_test/test_set.json
data/reva_test/videos/
data/reva_train/train_set.json
data/qwen_train/videos/
```

with the full instructor-provided ReVA annotations and matching videos. Then regenerate Qwen training data:

```bash
REVA_TRAIN_JSON=data/reva_train/train_set.json \
QWEN_TRAIN_JSON=data/qwen_train/train.json \
QWEN_VIDEO_ROOT=data/qwen_train \
MAX_SAMPLES=200 \
REQUIRE_VIDEO=1 \
bash scripts/prepare_qwen_train_data.sh
```
