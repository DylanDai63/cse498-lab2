# Unit Tests

Run from the project root:

```bash
<QWEN_ENV_PYTHON> -m pytest -q
```

At the beginning of the assignment these tests are expected to fail with `NotImplementedError`, because the `TODO(student)` functions are blank.

Test mapping:

- `test_build_qwen_train_data_helpers`: Part 1 prompt/answer formatting.
- `test_build_qwen_train_data_conversion`: Part 1 nested ReVA to Qwen sample conversion.
- `test_prepare_reva_v2_flatten`: Part 2 ReVA test flattening.
- `test_score_reva_predictions`: Part 2 answer parsing and accuracy computation.
- `test_vila_helpers`: Part 3 VILA instance loading, path resolution, prompt building, answer parsing, metrics.

Passing these tests does not replace the full workflow run, but it verifies the expected input/output behavior before GPU jobs.
