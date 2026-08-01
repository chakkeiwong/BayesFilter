# NeuTra HMC Program Phase A Local Check Record

Date: 2026-07-15  
Status: `PASS`

## Broad Focused Suite

CPU-only verification was deliberate and GPU devices were hidden before
TensorFlow import:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
  tests/test_neutra_hmc.py \
  tests/test_neutra_hmc_route_policy.py \
  tests/test_lgssm_neutra_gap_closure.py \
  tests/test_lgssm_neutra_robustness_s1.py \
  tests/test_lgssm_new_fixture_plain_hmc_f0.py \
  tests/test_lgssm_new_fixture_neutra_training_f1.py \
  tests/test_lgssm_new_fixture_neutra_hmc_f2.py \
  tests/test_neutra_training.py \
  tests/test_neutra_batching.py \
  tests/test_deterministic_lgssm_exact_target_tf.py \
  tests/test_tensorflow_gpu_memory_policy.py \
  tests/test_lgssm_neutra_training_tf.py \
  tests/test_neutra_gpu_bounded_training_tf.py
```

Result: `92 passed, 2 warnings in 34.09s`. Both warnings are the same
third-party TensorFlow Probability `distutils.version` deprecation warning;
there was no project failure or skip.

## Static And Artifact Checks

- `py_compile` passed for the shared core/policy, training/batching modules, all
  S1/F0/F1/F2 campaign modules, and their four CLIs.
- `git diff --check` passed for every Phase A repair.
- Both repaired JSON documents pass `python -m json.tool`.
- F0 fixture identity stable self-hash recomputed exactly as
  `sha256:1936ff2a29f46d60930ae2c02bde850da82ef364e63625922fac74f61ffabe56`.
- Active program-root integrity verifier checked 48 `path`/`file_sha256`
  references, 120 `tensor_path`/`tensor_file_sha256` references, 35 byte-count
  references, and 75 stable self-hashed JSON artifacts: `0 errors`.
- Route discovery/policy tests cover unledgered, stale, duplicate, missing-core,
  fixed-terminal, and reachable local-sampler bypass cases.
- Active source scan found TFP HMC construction only in the shared canonical
  core and no active NumPy or host callback.
- Post-close enforcement recheck: shared core, route policy, migrated campaign,
  and S1/F0/F1/F2 suites returned `42 passed, 2 warnings in 8.03s`; warnings
  were the same third-party TFP deprecations.
- Lazy public inference-export and common-runtime compatibility recheck returned
  `58 passed, 2 warnings in 3.95s`; warnings were the same third-party TFP
  deprecations.
- All five F1 terminal-manifest rows match both their result-file SHA-256 and
  embedded stable artifact hash.

No GPU job was rerun during terminal verification. GPU training validity comes
from the immutable S1/F1 run artifacts; the terminal suite intentionally tests
logic under CPU-hidden execution.
