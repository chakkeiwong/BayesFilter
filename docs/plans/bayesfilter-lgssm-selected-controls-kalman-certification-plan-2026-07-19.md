# LGSSM Selected-Control Kalman Certification Plan

Date: 2026-07-19

Status: `CLOSED_T10_INCONCLUSIVE_T50_SCORE_FAIL`

Result:
`docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-result-2026-07-19.md`

Campaign ID: `lgssm-selected-controls-kalman-certification-20260719`

Output root:
`docs/benchmarks/artifacts/lgssm_selected_controls_kalman_20260719/`

## Research Intent Ledger

| Field | Frozen intent |
| --- | --- |
| Main question | Do the independently tuned canonical Contract E--Chol LGSSM scopes at `T=10` and `T=50` agree with the exact differentiated Kalman likelihood in value and total score? |
| Candidate | `T=10`: `sinkhorn_steps=20,balance_steps=3`; `T=50`: `sinkhorn_steps=20,balance_steps=8`; both use `N=1024`, float32/TF32, GPU/XLA, `K=N`, and the exact claim seed blocks selected without Kalman output. |
| Tuning evidence | `T=10` selection under `ledh_offline_ot_cheaper_first_tuning_20260718/attempt01`; `T=50` selection under `ledh_per_scope_tuning_20260719/lgssm_t50_attempt02`. |
| Oracle | Float64 differentiated Kalman observed-data likelihood on the same dataset prefix and physical parameter point, transformed to the declared unconstrained HMC coordinates. |
| Expected failure mode | Marginal-valid finite transport may still have finite-particle value or score bias; source changes after tuning may also have changed the executed finite program. |
| Primary criterion | For each horizon, all engineering gates and exact tuning bindings pass; a simultaneous 95% Bonferroni-Student interval for relative value bias is contained in `[-0.001,0.001]`; and each of the five HMC-score relative-bias intervals is contained in `[-0.05,0.05]`. |
| Promotion veto | Nonfinite output, replay/chart/reset/marginal/work failure, wrong chunk policy, Python horizon unrolling, non-XLA execution, wrong controls/seeds, missing Kalman output, or failure to revalidate the selected controls on the original claim seeds under the current source. |
| Continuation veto | Either horizon is `screen_fail`, `inconclusive`, or engineering-invalid after one localized repair retry. Nonlinear transfer begins only after both horizons are `screen_pass`. |
| Explanatory diagnostics | Raw physical/HMC score differences, per-seed errors, average predictive-score OPG spectrum, RMS OPG error, maximum diagonal standardized error, compile/warm time, and allocator peak. |
| Forbidden conclusion | No parameter-region, posterior, HMC, nonlinear-model, universal-control, leaderboard-completeness, or method-superiority claim follows from this center-scoped test. |

## Evidence Contract

For estimator seed `s`, horizon `T`, exact Kalman value `L_K`, exact Kalman HMC
score `g_K`, candidate value `L_s`, and candidate HMC score `g_s`, define

```text
z[s,value] = (L_s - L_K) / abs(L_K)
z[s,j] = (g_s[j] - g_K[j]) / abs(g_K[j]).
```

Use the estimator seed as the uncertainty unit. For each horizon, form six
two-sided simultaneous 95% Bonferroni-Student intervals using 16 seeds and the
pre-existing critical value `3.036283222821165`. Classify a horizon as:

- `screen_pass` only when every interval is contained in its frozen region;
- `screen_fail` when any interval lies wholly outside its region or a hard
  engineering veto fires; and
- `inconclusive` otherwise.

Componentwise normalization is retained because it is the previously reviewed
LGSSM center screen and all five exact center score coordinates are nonzero.
It is not promoted to a nonlinear-model rule. The current average-OPG
diagnostic is emitted with `lambda=0`, `epsilon_0=1`, `epsilon_min=0`, and
`D=I` only as a secondary diagnostic. Those settings and outputs cannot change
the primary classification.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
| --- | --- | --- | --- |
| `(20,3)` at `T=10` | Frozen selected scope | Passed disjoint tuning and claim marginal gates without Kalman use | Wrong cross-horizon reuse is blocked by exact selection/seed binding |
| `(20,8)` at `T=50` | Frozen selected scope | Passed independent T=50 calibration, validation, and untouched claim | Wrong transfer from T=10 is blocked by the T=50 scope artifact |
| Claim seeds `81700..81715` and `81820..81835` | Existing untouched claim blocks | Directly answers whether the already accepted finite-program claims agree with Kalman | No new seed selection after oracle inspection |
| Uncached transport execution | Existing selected-control semantics | Avoids making the later optional dense geometry cache part of the scientific claim | Current source may still drift; rerun every original claim seed and require all original engineering gates |
| Float32/TF32 GPU/XLA | Production target and exact tuning scope | Tests the actual selected production-shaped route | FP64 agreement alone would not answer the question |
| 16 seeds and Student intervals | Existing owner-selected exploratory audit size | Supplies uncertainty rather than a one-seed comparison | No power guarantee; report inconclusive honestly |
| Value `0.1%`, score `5%` center margins | Existing reviewed LGSSM center criteria | Preserves the declared test rather than selecting a threshold after seeing new results | Not transferable to nonlinear models or HMC trajectory validity |

## Skeptical Plan Audit

Verdict: `PASS_AFTER_REPAIRING_BASELINE_AND_METRIC_DRIFT`.

- Wrong baseline: repaired by binding each node to its own selected-control
  artifact and original untouched claim, rather than using the obsolete common
  `balance_steps=2` or `50` runs.
- Proxy promotion: transport marginals remain hard engineering vetoes but do
  not establish oracle agreement; the Kalman intervals are the scientific
  criterion.
- Hidden source drift: the repository changed after tuning. The current source
  must rerun every original claim seed with the selected controls and pass all
  original engineering gates. Exact historical float32 equality is reported,
  but it is not a gate after an independently tested semantics-preserving
  evaluation-order optimization; no retrospective numerical tolerance is
  invented.
- Metric drift: average OPG is useful near zero but has no predeclared
  acceptance region. It remains explanatory and cannot retrospectively pass a
  failed historical center screen.
- Environment mismatch: runs require the trusted TensorFlow GPU/XLA path,
  float32/TF32, the 8192 MiB logical-device cap, and exact divisor chunking.
- Missing stop condition: nonlinear execution is forbidden unless both
  horizons pass; one localized harness repair retry is allowed without changing
  controls, seeds, margins, target, or hardware class.
- Artifact adequacy: fresh versioned node JSON plus one aggregate JSON and a
  result note preserve command, commit, source hashes, controls, seeds, device,
  timing, numerical diagnostics, and decision.

## Execution

1. Run focused aggregator and existing fused-runner tests CPU-only.
2. Run a trusted GPU/XLA `T=10` node with `(20,3)` and seeds `81700..81715`.
3. Run a trusted GPU/XLA `T=50` node with `(20,8)` and seeds `81820..81835`.
4. Aggregate both nodes against their frozen selections, stored claims, and the
   exact Kalman oracle.
5. If both screens pass, write the nonlinear `T=10,T=50` extension and
   per-model/per-horizon tuning plan. If either does not pass, diagnose the
   smallest implementation or scientific cause and do not claim nonlinear
   transfer.

## Budget And Stop Conditions

- One primary GPU node per horizon plus one localized repair retry per horizon.
- Total LGSSM campaign cap: 45 trusted GPU minutes.
- Stop on source-continuity failure that cannot be explained without changing
  the finite program, invalid Kalman identity, corrupted artifacts, exhausted
  retry/budget, or either final horizon not reaching `screen_pass`.
