# ReVA Training Annotations

The official dataset is available at [ReVA-Benchmark/ReVA](https://huggingface.co/datasets/ReVA-Benchmark/ReVA). Download it with:

```bash
export REVA_DATA_ROOT=/path/to/ReVA
hf download ReVA-Benchmark/ReVA --repo-type dataset --local-dir "$REVA_DATA_ROOT"
```

The downloaded root contains `train_set.json` and the video directories referenced by each annotation `file_path`. Preserve that directory structure.

Place ReVA training annotations here:

```text
data/reva_train/train_set.json
```

Generate Qwen-format training data with:

```bash
bash scripts/prepare_qwen_train_data.sh
```

The script requires referenced videos to exist by default and will not overwrite the current training JSON when no valid samples are found.

By default this creates a small 200-sample subset at:

```text
data/qwen_train/train.json
```

Use all available QA pairs with:

```bash
MAX_SAMPLES=0 bash scripts/prepare_qwen_train_data.sh
```
