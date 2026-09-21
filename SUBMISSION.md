# CSE 398/498 Lab 2 submission (Hengde Dai, hed424)

This repository is the assignment repository
[`likaiw2/VLM_evaluating_finetuing`](https://github.com/likaiw2/VLM_evaluating_finetuing)
(upstream commit `4623123`) with my work committed on top. `git diff 4623123 -- . ':!outputs' ':!data'`
shows every code change.

The report is `report/main.pdf` (source `report/main.tex`).

## Results (500-question stratified subset of the ReVA test split, seed 2026)

| Model | Accuracy | Questions | Answered |
| --- | ---: | ---: | ---: |
| Qwen3-VL-4B base | 65.20% | 500 | 495 |
| Qwen3-VL-4B fine-tuned (LoRA) | 68.40% | 500 | 500 |
| VILA1.5-3b base | 47.80% | 500 | 500 |

Base vs. fine-tuned on the same questions: 46 gained, 30 lost, exact McNemar p = 0.085.

## Deliverables

| Item | Path |
| --- | --- |
| Completed TODO code | `scripts/build_qwen_train_data.py`, `reva_eval/data/rsvidqa/prepare_reva_v2_test_set.py`, `scripts/score_reva_predictions.py`, `vila_eval/reva_v2.py` |
| Model comparison | `outputs/model_comparison.csv` |
| Qwen base `result.csv` | `outputs/qwen_base/qwen_base/result.csv` |
| Qwen fine-tuned `result.csv` | `outputs/qwen_sft/qwen_sft/result.csv` |
| VILA `metrics.json` | `outputs/vila_reva_v2/metrics.json` |
| Qualitative analysis (5 cases) | `report/main.pdf`, Section 6 |
| Commands | this file and `report/main.pdf`, Section 7 |

Also included, because the report numbers and the three examples are computed from them: the
per-question results (`result.json`, `outputs.jsonl`), the training log summary
(`outputs/qwen_reva_sft/trainer_state.json`), the 500-question evaluation subset and the 200 training
samples. `outputs/` is ignored by the upstream `.gitignore`, so these files were added with `git add -f`.

Not included: the ReVA videos (28 GB), model weights, the LoRA adapter weights, raw logs, and the conda
environments. They remain on the course server under `~/cse498_lab2/`.

## Unit tests

```bash
python -m pytest -q        # 7 passed (5 failed, 2 passed before the TODOs were completed)
```

## Files I added

| File | Purpose |
| --- | --- |
| `scripts/make_eval_subset.py` | Reproducible stratified evaluation subset plus the complementary "rest" file |
| `data/eval_subsets/test_subset_500_seed2026.json` | The 500-question subset used for all three models |
| `vila_eval/vila_local_patches.diff` | My three patches to `NVlabs/VILA@0f1426e` (`git apply` inside the VILA checkout) |
| `report/` | Report source, PDF and `make_results_tex.py`, which regenerates every number in the report |

## Hardware adaptation switches

The course server has RTX 2080 Ti GPUs (Turing, 11 GB): no FlashAttention 2, no native bfloat16.
Every switch is an environment variable. **Unset = original behaviour.**

| Variable | File | Default | Value I used |
| --- | --- | --- | --- |
| `ATTN_IMPLEMENTATION` | `qwen_finetune/qwenvl/train/train_qwen.py` | `flash_attention_2` | `sdpa` |
| `DATA_FLATTEN` | `qwen_finetune/scripts/sft_7b.sh` | `True` | `False` |
| `PRECISION` | `qwen_finetune/scripts/sft_7b.sh` | `bf16` | `none` |
| `MODEL_DTYPE` | `qwen_finetune/qwenvl/train/train_qwen.py` | follows `--bf16` | `bfloat16` |
| `EXTRA_TRAIN_ARGS` | `qwen_finetune/scripts/sft_7b.sh` | empty | see the fine-tuning command below |
| `EVAL_BATCH_SIZE` | `reva_eval/inference_vllm_origin_number.py` | `8` | `4` |
| `GPU_IDS` | `reva_eval/eval_reva_v2.sh` | `0 .. NUM_CHUNKS-1` | free GPUs on the shared server |

`qwen_finetune/qwenvl/train/trainer.py` also imports `flash_attn` optionally.

## Commands, in order

`$REVA` = ReVA root, `$QWEN` = Qwen3-VL-4B-Instruct directory, `$SUBSET` =
`data/eval_subsets/test_subset_500_seed2026.json`.

```bash
# Environment (qwen2)
conda create -n qwen2 python=3.10 -y && conda activate qwen2
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 pip install "transformers>=4.57.0,<4.58" accelerate==1.7.0 peft==0.17.1 \
    deepspeed==0.17.1 decord opencv-python-headless tqdm requests pytest torchcodec==0.2.1 av
conda install -n qwen2 -c conda-forge "ffmpeg>=6,<8" -y

# Data
hf download ReVA-Benchmark/ReVA --repo-type dataset --local-dir $REVA
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir $QWEN
for d in ERA_Select Hawk_UAV UAVDT VisDrone; do ln -sfn $REVA/$d data/qwen_train/$d; done
REVA_TRAIN_JSON=$REVA/train_set.json MAX_SAMPLES=200 bash scripts/prepare_qwen_train_data.sh
python scripts/make_eval_subset.py --input $REVA/test_set.json --size 500 --seed 2026

# Qwen baseline (the first run hit CUDA OOM at the default batch of 8 after 277 questions)
NUM_CHUNKS=4 EVAL_NAME=qwen_base REVA_ROOT=$REVA REVA_JSON=$SUBSET MODEL_PATH=$QWEN \
    bash scripts/run_eval_qwen_base.sh
RESUME=1 EVAL_BATCH_SIZE=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NUM_CHUNKS=4 \
    EVAL_NAME=qwen_base REVA_ROOT=$REVA REVA_JSON=$SUBSET MODEL_PATH=$QWEN \
    bash scripts/run_eval_qwen_base.sh

# LoRA fine-tuning (one hyperparameter changed: LR 2e-7 -> 1e-4)
CUDA_VISIBLE_DEVICES=3 ATTN_IMPLEMENTATION=sdpa DATA_FLATTEN=False PRECISION=none \
    MODEL_DTYPE=bfloat16 USE_DEEPSPEED=0 NPROC_PER_NODE=1 LR=1e-4 MODEL_PATH=$QWEN \
    EXTRA_TRAIN_ARGS='--ddp_find_unused_parameters False --gradient_checkpointing_kwargs {"use_reentrant":false}' \
    bash scripts/run_finetune_qwen.sh

# Fine-tuned Qwen
GPU_IDS="2 3" NUM_CHUNKS=2 EVAL_BATCH_SIZE=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EVAL_NAME=qwen_sft REVA_ROOT=$REVA REVA_JSON=$SUBSET MODEL_BASE=$QWEN \
    MODEL_PATH=outputs/qwen_reva_sft/checkpoint-50 bash scripts/run_eval_qwen_finetuned.sh

# VILA (environment: torch 2.3.0, transformers 4.46.0, flash_attn 2.5.8 wheel for import only,
# triton 3.1.0; patches in vila_eval/vila_local_patches.diff)
GPU=1 VILA_REPO=$VILA MODEL_PATH=$VILA_MODEL REVA_ROOT=$REVA REVA_JSON=$SUBSET \
    bash scripts/run_eval_reva_vila.sh

# Comparison and report numbers
bash scripts/compare_models.sh
python report/make_results_tex.py
```

## Extending to the full test split

`scripts/make_eval_subset.py` also writes `data/eval_subsets/test_rest_3500_seed2026.json`, the other
3500 questions with the same stable ids (same seed, deterministic). Evaluate it with the same settings
under a new `EVAL_NAME`, copy the prediction shards next to the existing ones, and score against the
full prepared test set. No question is evaluated twice. Then run `python report/make_results_tex.py`.
