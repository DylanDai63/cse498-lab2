# ReVA Test Data

The official dataset is available at [ReVA-Benchmark/ReVA](https://huggingface.co/datasets/ReVA-Benchmark/ReVA). Download it with:

```bash
export REVA_DATA_ROOT=/path/to/ReVA
hf download ReVA-Benchmark/ReVA --repo-type dataset --local-dir "$REVA_DATA_ROOT"
```

The downloaded root contains `test_set.json` and the corresponding video directories. Keep all paths relative to the ReVA root unchanged.

Place the ReVA evaluation subset here:

```text
data/reva_test/test_set.json
data/reva_test/<video files or extracted frame folders>
```

For the original server layout, this corresponds to:

```text
<REVA_DATA_ROOT>
```

The evaluation script can also read another location through:

```bash
REVA_ROOT=/path/to/ReVA_V2 REVA_JSON=/path/to/ReVA_V2/test_set.json bash scripts/run_eval_reva.sh
```
