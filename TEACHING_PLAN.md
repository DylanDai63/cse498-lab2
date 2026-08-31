# Teaching Plan: Qwen Fine-Tuning with ReVA and VILA Evaluation

## Course Goal

Students learn the complete loop for a visual language model experiment:

1. Prepare multimodal instruction data.
2. Fine-tune Qwen-VL with LoRA/SFT.
3. Evaluate the base and fine-tuned Qwen models on ReVA.
4. Evaluate VILA on the same ReVA split as a second model baseline.
5. Interpret quantitative scores and qualitative failure cases.

## Project Main Path

Use these files as the student-facing path:

```text
scripts/run_finetune_qwen.sh
scripts/run_eval_qwen_base.sh
scripts/run_eval_qwen_finetuned.sh
scripts/run_eval_reva_vila.sh
scripts/compare_models.sh
scripts/check_student_setup.py
qwen_finetune/scripts/sft_7b.sh
qwen_finetune/qwenvl/data/__init__.py
reva_eval/eval_reva_v2.sh
reva_eval/inference_vllm_origin_number.py
reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py
vila_eval/reva_v2.py
data/qwen_train/train.json
data/reva_test/test_set.json
```

## Teacher Preparation Checklist

- Download the official ReVA dataset from [ReVA-Benchmark/ReVA](https://huggingface.co/datasets/ReVA-Benchmark/ReVA).
- Download the Qwen checkpoint from [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), or pre-populate the shared Hugging Face cache.
- Put a small training set in `data/qwen_train/train.json`.
- Put matching training videos under `data/qwen_train/videos/`.
- Put ReVA `test_set.json` under `data/reva_test/test_set.json`.
- Put ReVA videos or extracted frame folders under `data/reva_test/`.
- Confirm the `qwen2` conda environment can run Qwen3-VL Transformers inference and training. The environment name is historical and does not indicate the model version.
- Confirm the `vila` conda environment can run VILA inference.
- Confirm `VILA_REPO` points to a working VILA checkout, such as `<VILA_REPO>`.
- Run baseline ReVA evaluation once and save reference output.
- Run VILA ReVA evaluation once and save reference output.
- Run one short fine-tuning job and save expected metric range.

Before publishing a release, validate the answer checkout with the exact public student tests:

```bash
ASSIGNMENT_PROJECT_ROOT=<ANSWER_PROJECT_ROOT> \
  <QWEN_ENV_PYTHON> -m pytest -q tests/test_student_todos.py
```

This prevents the student and answer repositories from drifting to different helper-function contracts.

## Suggested Assignment Tasks

### Task 1: Baseline Evaluation

Students run:

```bash
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash scripts/run_eval_qwen_base.sh
```

They report completion, total accuracy, and two subcategory accuracies from `result.csv`.

### Task 2: Fine-Tuning

Students inspect `data/qwen_train/train.json`, then run:

```bash
MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct bash scripts/run_finetune_qwen.sh
```

They submit the training command, changed hyperparameters, and final checkpoint path.

### Task 3: Fine-Tuned Evaluation

Students run:

```bash
MODEL_PATH=/path/to/fine-tuned-checkpoint bash scripts/run_eval_qwen_finetuned.sh
```

They compare baseline and fine-tuned scores.

### Task 4: VILA Baseline Evaluation

Students run:

```bash
VILA_REPO=<VILA_REPO> MODEL_PATH=Efficient-Large-Model/VILA1.5-3b bash scripts/run_eval_reva_vila.sh
```

They compare Qwen and VILA on the same ReVA questions.

### Task 5: Model Comparison and Error Analysis

Students run:

```bash
bash scripts/compare_models.sh
```

Students inspect `result.json` and select:

- two correct examples,
- two incorrect examples,
- one example where output formatting caused an error.

For each example, they explain whether the issue is visual perception, temporal reasoning, option parsing, or overfitting.

## Suggested Grading Rubric

| Item | Points |
| --- | ---: |
| Baseline ReVA evaluation runs successfully | 15 |
| Fine-tuning runs and produces a checkpoint | 25 |
| Fine-tuned ReVA evaluation runs successfully | 25 |
| VILA baseline evaluation runs successfully | 10 |
| Quantitative comparison is clear | 10 |
| Qualitative error analysis is specific | 15 |
| Commands and paths are reproducible | 5 |

## What to Exclude from the Student Version

Do not require students to use:

- `train_stage_2_dgrpo.sh`
- `train_stage_4_dgrpo.sh`
- multi-dataset MTVR training
- GPT-based grading
- unrelated VLM baselines
- raw HuggingFace cache directories
- server-specific paths such as `<USER_HOME>/...`
