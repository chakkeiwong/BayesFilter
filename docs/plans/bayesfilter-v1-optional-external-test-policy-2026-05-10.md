# Policy: BayesFilter v1 Optional External Tests

## Date

2026-05-10

## Purpose

BayesFilter should certify compatibility with MacroFinance and DSGE without
requiring either project as a production dependency or CI dependency before v1.
This policy defines how optional external tests are labeled, run, skipped, and
interpreted.

## External Test Classes

### Local CI Tests

Run inside BayesFilter with no external checkout.

Examples:

```text
tests/test_linear_kalman_qr_tf.py
tests/test_linear_kalman_qr_derivatives_tf.py
tests/test_linear_kalman_svd_tf.py
tests/test_structural_svd_sigma_point_tf.py
tests/test_svd_cut_filter_tf.py
tests/test_svd_cut_derivatives_tf.py
```

These are required for BayesFilter v1 development.

### Optional Live External Tests

Run only when an external checkout is present.

Current example:

```text
tests/test_macrofinance_linear_compat_tf.py
```

Expected behavior:

- skip clearly if `/home/chakwong/MacroFinance` is absent;
- import external modules only inside tests;
- never import external projects from BayesFilter production modules;
- record external checkout path and, when possible, commit hash in result
  notes;
- certify compatibility with the observed external state only.

### Deferred Integration Tests

These belong to a later v1 integration lane, not the current external
compatibility lane.

Examples:

```text
MacroFinance adapter branch tests
DSGE adapter branch tests
client default switch-over tests
```

## Recommended Commands

BayesFilter-local compatibility subset:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Optional live MacroFinance check:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_macrofinance_linear_compat_tf.py \
  -p no:cacheprovider
```

The live check should be reported separately from local CI.

## Interpretation Rules

- Local CI pass means BayesFilter's own v1 compatibility fixtures pass.
- Optional live pass means BayesFilter matches the observed external checkout on
  the tested fixtures.
- Optional live skip means the external checkout was unavailable; it is not a
  BayesFilter failure.
- Optional live failure is an external compatibility issue, not automatically a
  production regression.
- No optional live result authorizes client default switch-over.

## Stop Rules

Stop and ask for direction if an external compatibility test can pass only by:

- editing MacroFinance or DSGE source;
- importing client code into BayesFilter production modules;
- changing BayesFilter v1 API solely to match a client private helper;
- weakening diagnostics or masking conventions;
- changing tolerances without an artifact explaining why.
