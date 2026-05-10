# Result: BayesFilter v1 Post-completion Gap Closure

## Date

2026-05-11

## Scope

This result executes:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-plan-2026-05-11.md
```

It stays inside the BayesFilter v1 external-compatibility lane.  It does not
edit MacroFinance, DSGE, the shared monograph reset memo, structural SVD/SGU
plans, or Chapter 18/18b.

## Phase A/B: Plan Tightening And Audit

Execution:
- reread the plan and tightened Phase C so one-axis ladder claims require
  one-axis ladders;
- added independent audit:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-plan-audit-2026-05-11.md
```

- extended benchmark harness with:
  - `--benchmark-selector`;
  - `--modes`;
  - `--graph-warmup-calls`;
  - `--device-scope`;
  - `v1_time_ladder`;
  - `v1_parameter_ladder`;
  - `v1_state_observation_ladder`;
  - `v1_cpu_diagnostic`.

Interpretation:
- the plan is safe for scoped execution;
- GPU remains conditional on escalated probes;
- external project work remains read-only/optional.

## Phase C: CPU QR Derivative Diagnostic Ladders

Artifacts:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-11-smoke.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-11-smoke.md
docs/benchmarks/bayesfilter-v1-filter-benchmark-ladder-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-ladder-2026-05-11.md
docs/benchmarks/bayesfilter-v1-qr-score-hessian-time-ladder-2026-05-11.json
docs/benchmarks/bayesfilter-v1-qr-score-hessian-time-ladder-2026-05-11.md
docs/benchmarks/bayesfilter-v1-qr-score-hessian-parameter-ladder-2026-05-11.json
docs/benchmarks/bayesfilter-v1-qr-score-hessian-parameter-ladder-2026-05-11.md
```

Observed:
- smoke benchmark: all rows `status = ok`;
- mixed CPU diagnostic ladder: all rows `status = ok`;
- time ladder: all rows `status = ok`;
- parameter ladder: all rows `status = ok`.

Interpretation:
- H1 is supported.  QR score/Hessian cost is dominated by graph warmup/tracing
  and materialization, not steady recurrence cost.  After one graph warmup,
  measured calls are millisecond-scale.
- H2 is partially supported.  Parameter dimension and time dimension are both
  strong cost drivers.

Key numbers:
- time ladder, fixed `state_dim=2`, `observation_dim=2`, `parameter_dim=2`:
  - `timesteps=4`: warmup about 11.3 seconds, max RSS delta about 638.3 MB;
  - `timesteps=16`: warmup about 35.4 seconds, max RSS delta about 1702.1 MB.
- parameter ladder, fixed `timesteps=8`, `state_dim=2`,
  `observation_dim=2`:
  - `parameter_dim=2`: warmup about 20.4 seconds, max RSS delta about
    1236.8 MB;
  - `parameter_dim=4`: warmup about 64.3 seconds, max RSS delta about
    3319.8 MB.

Audit:
- the state/observation ladder was not run because the parameter ladder pushed
  process high-water RSS above 7 GB;
- continuing into larger ladders would add machine pressure without changing
  the main conclusion.

## Phase D: GPU/XLA-GPU Evidence

Escalated probes:

```text
nvidia-smi: NVIDIA GeForce RTX 4080 SUPER visible, 16376 MiB total memory.
TensorFlow 2.19.1 physical devices: CPU and GPU visible.
```

Artifact:

```text
docs/benchmarks/bayesfilter-v1-filter-benchmark-gpu-visible-smoke-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-gpu-visible-smoke-2026-05-11.md
docs/benchmarks/bayesfilter-v1-filter-benchmark-xla-visible-2026-05-11.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-xla-visible-2026-05-11.md
```

Observed:
- all GPU-visible smoke rows `status = ok`;
- TensorFlow logical devices include CPU and GPU.
- XLA-visible linear value rows `status = ok`.

Interpretation:
- GPU availability is proven under escalation;
- small-shape GPU-visible execution works;
- XLA-visible linear value rows complete in a GPU-visible process;
- no broad GPU speedup, confirmed GPU placement for every XLA op, QR
  derivative XLA, or HMC readiness claim is justified.

## Phase E: Optional Live MacroFinance

External checkout:

```text
/home/chakwong/MacroFinance
0e81988957ef1f8b520014929bea32ffee3881f4
```

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_macrofinance_linear_compat_tf.py \
  -p no:cacheprovider
```

Observed:

```text
4 passed, 2 warnings in 74.54s (0:01:14)
```

Interpretation:
- optional live MacroFinance compatibility passes read-only on the observed
  checkout;
- the MacroFinance checkout was dirty with unrelated local edits, so this is
  live compatibility evidence on an observed checkout, not clean release
  certification;
- no MacroFinance files were edited;
- this does not authorize MacroFinance default switch-over.

## Phase F: DSGE Optional-fixture Design

Artifact:

```text
docs/plans/bayesfilter-v1-dsge-test-only-fixture-design-2026-05-11.md
```

Decision:
- Rotemberg is a future optional live structural fixture candidate;
- EZ is a future metadata-only optional fixture candidate;
- SGU remains blocked by causal locality.

## Phase G: First HMC Target

Artifact:

```text
docs/plans/bayesfilter-v1-hmc-first-target-selection-2026-05-11.md
```

Decision:

```text
linear_qr_score_hessian_static_lgssm
```

Rationale:
- QR derivative evidence is strongest;
- optional live MacroFinance compatibility passed;
- SVD-CUT and DSGE HMC remain blocked by separate gates.

## Phase H: SVD-CUT Branch Diagnostic Gate

Artifact:

```text
docs/plans/bayesfilter-v1-svd-cut-branch-diagnostic-gate-2026-05-11.md
```

Decision:
- SVD-CUT derivative claims require smooth separated spectrum, inactive floors,
  and finite-difference or autodiff parity;
- HMC remains blocked until target-specific branch-frequency and sampler
  evidence exist.

## Phase I: CI Runtime Tier Policy

Artifact:

```text
docs/plans/bayesfilter-v1-ci-runtime-tier-policy-2026-05-11.md
```

Decision:
- split evidence into fast local CI, focused local regression, extended CPU
  diagnostics, optional live external, escalated GPU/XLA-GPU, and HMC readiness
  tiers.

## Hypothesis Status

| Hypothesis | Status |
| --- | --- |
| H1 QR cost is mostly tracing/materialization | supported |
| H2 parameter/time dimensions drive cost | supported for parameter and time; state/observation not run due memory caution |
| H3 benchmark harness supports diagnostic ladders | supported |
| H4 escalated TensorFlow can see GPU | supported |
| H5 optional live MacroFinance passes read-only | supported |
| H6 Rotemberg can be designed as test-only fixture | design closed |
| H7 EZ can be metadata-only optional fixture | design closed |
| H8 first HMC target should be QR derivative | accepted |
| H9 SVD-CUT promotion blocked by branch diagnostics | accepted |
| H10 linear SVD/eigen derivatives unnecessary for v1 | remains deferred |
| H11 CI can be tiered | policy closed |
| H12 no other-agent files needed | supported |

## Next Suggestions

1. Implement a QR derivative memory-reduction experiment that isolates Hessian
   materialization and parameter-major contractions.
2. Add an extended CPU diagnostic that records QR derivative graph build cost
   with state/observation ladder only after memory pressure is acceptable.
3. Build the first HMC smoke target for
   `linear_qr_score_hessian_static_lgssm`.
4. Add SVD-CUT branch-frequency aggregation over a small parameter box before
   any SVD-CUT HMC experiment.
5. Keep MacroFinance switch-over deferred until a separate integration lane
   records rollback and default-change criteria.

## Final Validation

Commands run before commit:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  -p no:cacheprovider

35 passed, 2 warnings in 92.35s (0:01:32)
```

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 1 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2 \
  --benchmark-selector linear_value \
  --modes graph

all rows status = ok
```

Additional validation:
- benchmark JSON syntax checks passed for the CPU smoke, GPU-visible smoke, and
  XLA-visible value artifacts;
- `git diff --check` passed;
- staged-file audit must include only v1-lane files.

## Completion Interpretation

The post-completion gap-closure phase is complete for this lane.  It closes the
QR derivative cost diagnosis as a measured warmup/materialization problem,
records narrow GPU-visible and XLA-visible evidence, upgrades optional
MacroFinance status from `not_run_by_policy` to read-only pass on the observed
checkout, designs DSGE optional fixtures without editing DSGE, selects a first
HMC target, and preserves all blocked claims that still lack target-specific
evidence.

## Final Validation And Commit Boundary

Final validation:
- `git diff --check`: passed;
- `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q
  tests/test_v1_public_api.py -p no:cacheprovider`: `2 passed,
  2 warnings`;
- `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py --repeats 1
  --benchmark-selector linear_value --modes graph --graph-warmup-calls 1
  --timesteps 4 --state-dim 2 --observation-dim 2 --parameter-dim 2
  --output /tmp/bayesfilter-v1-final-smoke.json --markdown-output
  /tmp/bayesfilter-v1-final-smoke.md`: all rows `status = ok`.

Commit boundary:
- stage only v1 external-compatibility files under `docs/benchmarks`,
  `docs/plans/bayesfilter-v1-*`, and `docs/source_map.yml`;
- do not stage the shared monograph reset memo, structural SVD/SGU work,
  Chapter 18/18b files, external project files, template sidecars, local
  images, or unrelated DSGE request files.
