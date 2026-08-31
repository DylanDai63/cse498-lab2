# Runnable Instructions for Students

Run everything from:

```bash
cd <PROJECT_ROOT>
export PATH=<CONDA_ROOT>/bin:$PATH
export MODEL_PATH=<QWEN3_VL_MODEL_PATH>
```

The prepared teaching environment is named `qwen2` for historical reasons, but this assignment consistently uses the Qwen3-VL-4B-Instruct model.

## 0. Download Official Resources

- ReVA dataset: [ReVA-Benchmark/ReVA](https://huggingface.co/datasets/ReVA-Benchmark/ReVA)
- Qwen model weights: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

If the instructor has not already placed them on the server, download them with:

```bash
python3 -m pip install -U huggingface_hub

export REVA_DATA_ROOT=/path/to/ReVA
export QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct

hf download ReVA-Benchmark/ReVA \
  --repo-type dataset \
  --local-dir "$REVA_DATA_ROOT"

hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$QWEN3_VL_MODEL_PATH"
```

The ReVA download is about 29.9 GB and must retain its original directory structure. Alternatively, set `MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct` to let Transformers download and cache the model automatically.

## 1. Prepare Demo Data

The bundled project contains one demo video. This command creates distinct training and test questions that require reading different text from the video title card, so the full workflow can run before the complete ReVA videos are available.

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

The Qwen result reports `Completed` separately from `Total` accuracy. Missing predictions remain in the accuracy denominator.

For another fresh baseline run, choose a new output identity, for example `EVAL_NAME=qwen_base_run2`. Use `RESUME=1` only to continue the exact same model and evaluation configuration; mismatched resumes are rejected.

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

The demo training and test questions are distinct, but they still use one short video and are only for checking that code, environments, and commands work. Demo metrics are not valid assignment results. For the real assignment, replace:

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
