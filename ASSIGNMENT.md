# CSE Homework: Qwen3-VL Fine-Tuning and ReVA Evaluation

This repository is the student version of the Qwen3-VL/ReVA assignment. The infrastructure scripts, model paths, conda environments, and demo data are already prepared. Your job is to complete the missing core logic marked with `TODO(student)`.

Run all commands from:

```bash
cd <STUDENT_PROJECT_ROOT>
export PATH=<CONDA_ROOT>/bin:$PATH
```

The teacher reference implementation is in:

```text
<PROJECT_ROOT>
```

Do not edit the teacher reference directory.

## Part 1: Convert ReVA Annotations to Qwen Training Data

File: `scripts/build_qwen_train_data.py`

Complete `format_options`, `format_question`, `format_answer`, `iter_reva_qas`, and `convert_item`.

Test:

```bash
bash scripts/setup_demo_data.sh
python3 -m json.tool data/qwen_train/train.json | head -80
```

## Part 2: Prepare and Score ReVA Evaluation Data

Files:

```text
reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py
scripts/score_reva_predictions.py
```

Complete path normalization, QA flattening, answer extraction, letter parsing, prediction loading, and accuracy computation.

Test:

```bash
CONDA_ENV=qwen2 MODEL_PATH=<QWEN3_VL_MODEL_PATH> BACKEND=transformers MAX_FRAMES=4 bash scripts/run_eval_qwen_base.sh
```

## Part 3: VILA Baseline and Model Comparison

File: `vila_eval/reva_v2.py`

Complete `load_instances`, `resolve_video_path`, `build_prompt`, `parse_choice`, and `summarize`.

Test:

```bash
CONDA_ENV=vila VILA_REPO=<VILA_REPO> MODEL_PATH=Efficient-Large-Model/VILA1.5-3b MAX_QUESTIONS=2 NUM_VIDEO_FRAMES=4 bash scripts/run_eval_reva_vila.sh
```

## Part 4: Fine-Tune Qwen3-VL with LoRA

After Part 1 works, run:

```bash
CONDA_ENV=qwen2 MODEL_PATH=<QWEN3_VL_MODEL_PATH> NPROC_PER_NODE=1 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 EPOCHS=1 SAVE_STEPS=1 MAX_PIXELS=50176 VIDEO_MAX_FRAMES=4 VIDEO_FPS=1 MODEL_MAX_LENGTH=2048 USE_DEEPSPEED=0 bash scripts/run_finetune_qwen.sh
```

Then evaluate the LoRA checkpoint:

```bash
CONDA_ENV=qwen2 MODEL_PATH=outputs/qwen_reva_sft/checkpoint-2 MODEL_BASE=<QWEN3_VL_MODEL_PATH> BACKEND=transformers MAX_FRAMES=4 bash scripts/run_eval_qwen_finetuned.sh
```

## Part 5: Compare Results

```bash
bash scripts/compare_models.sh
cat outputs/model_comparison.csv
```

## Submission

Submit completed code, `outputs/model_comparison.csv`, Qwen base `result.csv`, Qwen fine-tuned `result.csv`, VILA `metrics.json`, and a short analysis of 3 examples.

## Unit Tests

Run the unit tests before running GPU jobs:

```bash
cd <STUDENT_PROJECT_ROOT>
<QWEN_ENV_PYTHON> -m pytest -q
```

At the beginning of the assignment these tests are expected to fail with `NotImplementedError`, because the `TODO(student)` functions are blank. After you complete the TODOs, all unit tests should pass before you run the Qwen/VILA GPU workflows.

These tests use tiny in-memory examples and do not load Qwen or VILA. They verify the expected input/output behavior of each TODO function.
