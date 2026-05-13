# Result: BayesFilter v1 External-compatibility Phase Completion

## Date

2026-05-11

## Scope

This result executes:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-2026-05-11.md
```

It stays inside the BayesFilter v1 external-compatibility lane.  It does not
edit MacroFinance, DSGE, the shared monograph reset memo, or Chapter 18/18b.

## Phase A: Lane Boundary And Plan Audit

Plan:
- confirm the lane boundary before execution;
- audit the phase plan as another developer;
- strengthen the plan if a blocker is found.

Execute:
- Confirmed current lane-owned files are:
  - `docs/plans/bayesfilter-v1-*.md`;
  - v1 benchmark files under `docs/benchmarks`;
  - `tests/test_v1_public_api.py`;
  - `docs/source_map.yml` entries whose keys begin with `bayesfilter_v1_`.
- Left the shared monograph reset memo out of scope.
- Added the independent audit:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-audit-2026-05-11.md
```

Audit result:
- The plan is approved for scoped execution.
- Required strengthening: Phase D must add memory metadata before the benchmark
  gap can close.

Next phase justified?
- Yes.  The audit found no lane-boundary veto.

## Phase B: Public API Import Gate

Plan:
- verify that declared v1 public symbols are top-level importable;
- verify importing `bayesfilter` does not import MacroFinance or DSGE modules.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Observed:

```text
2 passed, 2 warnings in 3.52s
```

Interpretation:
- The top-level v1 public API gate passes.
- The only warnings are TensorFlow Probability `distutils` deprecation
  warnings.

Next phase justified?
- Yes.  No client imports or unstable promoted symbols were detected.

## Phase C: Local Compatibility Regression Gate

Plan:
- run the focused BayesFilter-local regression subset under deliberate CPU-only
  settings.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Observed:

```text
31 passed, 2 warnings in 84.50s (0:01:24)
```

Interpretation:
- The local API, QR value, QR score/Hessian, SVD value, and compiled parity
  gates pass without MacroFinance or DSGE checkouts.

Next phase justified?
- Yes.  Local v1 compatibility evidence is sufficient to run benchmark
  hardening.

## Phase D: Benchmark Harness Hardening

Plan:
- add process-memory metadata to the benchmark harness;
- regenerate the smoke artifact;
- run a medium-shape CPU artifact if runtime remains reasonable.

Execute:
- Updated:

```text
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
```

- Added process-level RSS and high-water RSS fields to each benchmark row:
  - `rss_before_mb`;
  - `rss_after_mb`;
  - `rss_delta_mb`;
  - `max_rss_before_mb`;
  - `max_rss_after_mb`;
  - `max_rss_delta_mb`.
- Added Markdown artifact generation through `--markdown-output`.

Smoke command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 2 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2 \
  --output docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.md
```

Smoke result:
- all rows completed with `status = ok`;
- all rows include timing, shape, point-count, and memory metadata;
- artifact remains CPU-only and smoke-scale.

Medium command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 2 \
  --timesteps 12 \
  --state-dim 4 \
  --observation-dim 3 \
  --parameter-dim 3 \
  --output docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.json \
  --markdown-output docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.md
```

Medium result:
- all rows completed with `status = ok`;
- the QR score/Hessian first eager call took about 57.6 seconds and increased
  process high-water RSS by about 3335.8 MB;
- graph QR score/Hessian first call took about 11.1 seconds and increased
  high-water RSS by about 486.3 MB;
- steady calls were much faster, but the first-call compile/tracing and memory
  costs are real and should shape future performance work.

Audit:
- TensorFlow printed CUDA plugin-registration and `cuInit` messages even with
  `CUDA_VISIBLE_DEVICES=-1`.  The logical device list is CPU-only, so these
  runs are CPU evidence only and not GPU evidence.
- Memory fields are process-level diagnostics and are not isolated allocation
  profiles.

Next phase justified?
- Yes.  Benchmark metadata and medium-shape evidence now exist.  GPU remains a
  separate escalated-device phase.

## Phase E: Optional External Check Decision

Plan:
- decide whether optional live MacroFinance should run in this pass.

Decision:

```text
optional_live_macrofinance_status = not_run_by_policy
```

Interpretation:
- This phase deliberately avoids client coupling.
- The optional live MacroFinance test remains available:

```text
tests/test_macrofinance_linear_compat_tf.py
```

- Not running it is not a BayesFilter CI failure.

Next phase justified?
- Yes.  The optional external status is explicit.

## Phase F: DSGE Candidate Containment

Plan:
- keep the DSGE read-only inventory result as the current evidence;
- do not implement Rotemberg/EZ bridges;
- keep SGU blocked.

Evidence:

```text
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-2026-05-10.md
```

Interpretation:
- SGU remains blocked for production filtering.
- Rotemberg is the best future optional live DSGE fixture.
- EZ remains metadata/stability-only without BK/QZ/HMC certification.
- No BayesFilter production DSGE adapter is justified in this phase.

Next phase justified?
- Yes.  DSGE candidates remain contained.

## Phase G: GPU/HMC/SVD-derivative Blocker Review

GPU/XLA-GPU:
- blocked until escalated `nvidia-smi`, escalated TensorFlow device probe, and
  matching GPU benchmark artifacts exist.

HMC:
- blocked until a named target model has value/score/Hessian diagnostics,
  active-floor and spectral-gap diagnostics, compiled parity, and sampler
  evidence.

Linear SVD/eigen derivatives:
- deferred until a real client need proves QR derivatives are insufficient and
  a spectral-gap/floor policy is tested.

Next phase justified?
- Yes.  All blocked claims remain blocked with explicit evidence requirements.

## Phase H: Tidy And Commit Boundary

Pre-commit requirements:
- stage only v1-lane files;
- leave the shared monograph reset memo unstaged;
- leave unrelated `Zone.Identifier` files, local images, and DSGE request notes
  unstaged;
- run `git diff --cached --check`;
- commit scoped v1 artifacts.

## Summary

The phase is complete from an execution perspective:

- API gate: passed.
- Local regression gate: passed.
- Benchmark smoke artifact: passed with memory metadata.
- Medium CPU artifact: passed with memory metadata.
- Optional MacroFinance: not run by policy.
- DSGE: contained as read-only external evidence.
- GPU/HMC/SVD-derivative claims: explicitly blocked/deferred.

## Suggested Next Hypotheses

H1. QR score/Hessian first-call memory can be reduced by restructuring graph
construction or separating derivative contractions without changing the
implemented likelihood law.

H2. A larger medium CPU ladder can identify the shape at which QR
score/Hessian first-call memory becomes unacceptable for ordinary development
machines.

H3. Escalated GPU/XLA-GPU probes plus matching benchmark shapes will determine
whether XLA materially improves steady-state filtering time despite first-call
compile costs.

H4. Rotemberg can become the first DSGE optional live fixture if a test-only
bridge validates structural metadata, deterministic completion residuals, and
value likelihood without importing DSGE economics into BayesFilter production
modules.

H5. HMC readiness should first be tested on a named QR-derivative target before
any SVD/CUT derivative branch is promoted, because QR derivative evidence is
currently much stronger.
