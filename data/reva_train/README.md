# ReVA Training Annotations

Place ReVA training annotations here:

```text
data/reva_train/train_set.json
```

Generate Qwen-format training data with:

```bash
bash scripts/prepare_qwen_train_data.sh
```

By default this creates a small 200-sample subset at:

```text
data/qwen_train/train.json
```

Use all available QA pairs with:

```bash
MAX_SAMPLES=0 bash scripts/prepare_qwen_train_data.sh
```
