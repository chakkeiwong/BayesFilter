# BayesFilter DZ5 Structured-Curvature and Consensus-Repair Result

Date: 2026-07-16

Status: `ENGINEERING_IMPLEMENTATION_VERIFIED_NO_HMC_RUN`

Plan: `docs/plans/bayesfilter-dz5-structured-curvature-consensus-repair-plan-2026-07-16.md`

Implementation checkpoint: `56b1f3c` (`Add structured score-curvature initialization`), pushed to `origin/main` before this execution phase.

## Outcome

The ten-part engineering plan was implemented and verified. BayesFilter now has
a fixed-center score-curvature API that keeps nonzero center score separate from
curvature stability; repeated dense and structured fits; disjoint training,
selection, and post-selection audit partitions; weak-direction-sensitive
stability diagnostics; explicit one-factor-to-two-factor escalation; convex
consensus shrinkage with structured or diagonal targets; fail-closed fallback;
a `diagnostic_center` mass-artifact handoff; and an explicit spawn-based CPU/XLA
cloud evaluator with continuous semantic progress telemetry.

This closes the common engineering layer only. It does not run the MacroFinance
DZ5 target, establish a MAP, validate the both-binding geometry, run an HMC
mechanics canary, retain posterior chains, or establish convergence.

## Review disposition

The required bounded Claude Code review used `claude-sonnet-5`, read only the
plan, and returned `VERDICT: REVISE` after about 427 seconds. All material
findings were accepted. The revised plan:

- distinguishes covariance parameterization from precision prediction and HMC handoff;
- provides a disjoint `2N` post-selection audit cloud;
- makes stability eligibility gates operational and treats missing caps as diagnostic-only;
- derives the `2N` and `3N-1` parameter counts;
- replaces “Ledoit--Wolf-style” with plain convex shrinkage;
- defines branch-specific fallback failure semantics;
- defines physical-core detection and the one-third worker rule;
- scopes target-only and child-process XLA evidence;
- adds factor-representation invariance coverage;
- separates GPU allocator configuration checks from GPU-readiness claims.

Claude session provenance is
`/home/ubuntu/.claude/projects/-home-ubuntu-python-BayesFilter/7ff24ff5-0943-43a2-a597-301fa62f8ffb.jsonl`.
That local transcript contains the exact prompt and response; it is not part of
the repository artifact.

## Skeptical post-implementation audit

The first green implementation was not accepted at face value. The second audit
found four material defects and stopped execution until each was repaired:

| Defect | Why the initial pass was misleading | Repair and discriminator |
| --- | --- | --- |
| Copied rows could cross partitions | Array alias checks did not detect equal rows in separate copies; the first synthetic fixtures used copied clouds | Reject shared-memory and exact float64 offset-row overlap across every training, selection, and audit partition; use distinct seeds; add alias and copied-row negative tests |
| Two factors were fit unconditionally | This contradicted one-before-two escalation and wasted exact-score fit work | Fit factor two only after one-factor fit, holdout, or stability failure, or an explicit factor-two target; test both skip and escalation branches |
| Structured shrinkage was unreachable | Direct stable-factor return shadowed an explicitly requested structured target | Explicit `structured_target_family` bypasses direct-factor return and enters shrinkage selection; add a selected-weight branch test |
| Child CPU hiding occurred too late | Spawn could import `bayesfilter.inference` and TensorFlow while unpickling before its initializer; CPU-hidden pytest masked the error | Move initializer/evaluator to lightweight `bayesfilter.cpu_xla_worker_bootstrap`, inherit CPU/growth settings before spawn re-import, restore parent environment, return child provenance, and fail closed on missing inherited settings |

The last process test imported TensorFlow in the parent, set
`CUDA_VISIBLE_DEVICES=parent-marker`, launched the pool without an outer
CPU-hiding environment, verified the parent marker was restored, and required
every child to report inherited `CUDA_VISIBLE_DEVICES=-1`, growth enabled,
`B=1`, and XLA compilation. A direct bad-environment test exercised the
fail-closed initializer.

## Implemented contracts

- `bayesfilter/inference/fixed_center_curvature.py` fits
  `g_z(c)-g_z(c+z) ~= P_z z` without moving or stationarity-gating `c`.
- Dense fits retain raw symmetric precision, eigenvalues, inertia, SPD
  projection, and projection burden.
- Structured fits parameterize `C_z=D[diag(1-||L_i||^2)+LL^T]D` and predict
  with its Cholesky-inverted precision `P_z=C_z^{-1}`.
- One-factor sign and two-factor triangular/sign identification are explicit;
  a prediction-Jacobian rank gate rejects unidentified factor two.
- Pairwise diagnostics include generalized-eigenvalue spread,
  trace-normalized Frobenius/operator differences, and principal angles.
- A separate audit cloud is touched after family, target, and shrinkage-weight
  selection and can veto but cannot retune.
- Fallback order is stable factor one, stable factor two, admissible dense
  consensus shrunk toward an explicitly requested stable factor target,
  admissible dense consensus shrunk toward its diagonal, explicit
  diagnostic-only diagonal, then blocked. Identity is never manufactured.
- Eligible results alone construct `PrecomputedMassArtifact` with
  `position_role="diagnostic_center"`.
- `bayesfilter/inference/cpu_xla_cloud.py` is an explicit CPU route, never an
  automatic GPU fallback. It uses physical cores when available, defaults to
  `floor(cores/3)`, supports override, clamps to the first task count, retains a
  persistent pool, restores input order, propagates child failure, and records
  semantic progress separately from liveness heartbeat.

## Requirement-to-test matrix

| Requirement | Test evidence | What it does not establish |
| --- | --- | --- |
| Variances, factor correlation identity, strict SPD | `test_one_factor_covariance_has_declared_variances_and_correlations`; `test_two_factor_covariance_matches_latent_factor_identity` | Empirical model adequacy |
| Row-ball interior/boundary | covariance construction tests and `test_factor_covariance_rejects_loading_row_ball_boundary` | Calibration of the `1e-6` margin |
| Factor sign/anchors and parameter counts | one/two-factor recovery tests | Global optimizer uniqueness |
| Degenerate Jacobian rejection | dimensional rejection plus `test_two_factor_fit_rejects_rank_deficient_prediction_jacobian` | DZ5 identification |
| Equivalent factor representation | `test_factor_covariance_is_invariant_to_column_rotation_and_permutation` | Optimizer invariance across every anchor choice |
| Covariance/precision orientation | factor recovery, fixed-center recovery, and diagnostic-center covariance scaling tests | Posterior covariance truth |
| `N=1,10,11` search boundaries and positivity | `test_dimension_scaled_search_rule_is_even_and_matches_boundaries` | Sample-complexity optimality |
| Reuse radius, nonzero, finite filtering | structured reuse integration and `test_reuse_filter_includes_radius_boundary_and_rejects_zero_outside_nonfinite` | Statistical independence of generator streams |
| No training/selection/audit overlap | `test_partition_views_fail_closed_and_audit_budget_is_enforced` | Independence beyond exact-row equality; lineage seeds remain required |
| One-to-two escalation | sequential escalation test plus fixed-center skip/escalate tests | Two factors are sufficient for DZ5 |
| Raw curvature/projection/stability | low-score instability and geometry-metric tests | Universal stability thresholds |
| Stable nonzero-score center | `test_stable_fixed_center_curvature_accepts_nonzero_score` | MAP attainment |
| Fresh audit veto and no retuning | `test_fresh_audit_cloud_can_veto_without_changing_selection` | Downstream HMC mechanics |
| Missing stability caps fail closed | `test_incomplete_stability_thresholds_remain_diagnostic_only` | Threshold calibration |
| Consensus SPD, structured target, diagonal status | consensus parity, structured-target, and diagonal-only selector tests | Classical Ledoit--Wolf theory or optimal shrinkage |
| Mass-artifact role and scaling | `test_diagnostic_center_artifact_uses_covariance_scaling_and_role`; HMC mass tests | HMC readiness or convergence |
| In-process XLA value/gradient | factor covariance XLA and consensus XLA gradient-parity tests | Full-chain XLA |
| Spawned child XLA/import order | CPU/XLA persistence/order test and bootstrap fail-closed test | CPU/GPU speed ranking or arbitrary third-party `__main__` behavior |
| Worker rule, override, task clamp | default-count and one-row pool tests | Performance-optimal worker count |
| Exception/lifecycle/order | pool initialization failure and repeated ordered evaluation tests | Supervisor recovery after OS-level worker death |
| Semantic progress versus heartbeat | row completion sequence and forced-heartbeat test | A universal timeout policy |
| GPU allocator configuration | `tests/test_common_inference_runtime_contracts.py` | Runtime GPU readiness; no GPU run occurred |
| Public imports and serialization | `tests/test_v1_public_api.py`, fixed-center payload JSON, CPU payload, mass tests | API stability beyond the current release |

All tests above are deterministic or fixed-seed engineering checks. They are
not stochastic method comparisons, so no ranking or superiority inference is
made.

## Verification record

Final relevant suite:

```bash
env -u CUDA_VISIBLE_DEVICES \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-pycache \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_factor_correlation_geometry.py \
  tests/test_sequential_map_covariance.py \
  tests/test_fixed_center_curvature.py \
  tests/test_cpu_xla_cloud.py \
  tests/test_common_inference_runtime_contracts.py \
  tests/test_hmc_mass_matrix.py \
  tests/test_v1_public_api.py
```

Result: `123 passed`, `342` dependency deprecation warnings, pytest time
`37.01s`, wall time `38.93s`. No test failed or skipped. The warnings come from
TensorFlow Probability `distutils` version checks and `gast` under Python 3.13;
they do not indicate a lane failure.

The exact selectively staged Git index was also exported to a clean temporary
tree and the same suite was rerun there. The repository's generated, ignored
`bayesfilter/ops/_symmetric_sylvester_ops.so` was copied into that tree because
the broad public-API test imports the custom op. With that existing build
prerequisite supplied, the staged tree produced `123 passed`, `342` warnings in
`38.95s`. This verifies that the selective commit does not depend on the other
agents' unstaged source changes.

Focused parent-independent process boundary:

```bash
env -u CUDA_VISIBLE_DEVICES \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-pycache \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest \
  -p no:cacheprovider -q tests/test_cpu_xla_cloud.py
```

Result after the final fail-closed assertion: `5 passed` in `12.76s`.

Environment: Python `3.13.13`, TensorFlow `2.20.0`, TensorFlow Probability
`0.25.0`, NumPy `2.1.3`, and `latexmk 4.76`.

## Math and document audit

`codex mcp list` reported no configured direct MCP server. The required CLI
fallback succeeded:

```bash
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/anaconda3/bin/python -m mathdevmcp.cli doctor
```

The doctor returned `ok: true`. Four `compare-label-code` checks returned
`status: consistent`, no missing required terms:

- `eq:bf-fixed-center-score-curvature` against `fixed_center_curvature.py`;
- `eq:bf-factor-correlation-covariance` against `factor_correlation_geometry.py`;
- `eq:bf-consensus-shrinkage` against `fixed_center_curvature.py`;
- `eq:bf-dimension-scaled-cloud` against `sequential_map_covariance.py`.

These checks establish term-level math/code correspondence, not formal proof.
Manual review also verified covariance/precision direction, scaling, signs,
parameter counts, audit timing, and fallback order.

The LaTeX build command was:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

It succeeded. `docs/main.pdf` has 398 pages and is 1,573,168 bytes. No new
label/reference or lane-local overfull warning remains. The build still reports
11 pre-existing undefined citations and four pre-existing multiply defined
labels in unrelated OT chapters.

`docs/source_map.yml` now parses as date `2026-07-16` with 80 sources. During
execution, two pre-existing historical source records were found incorrectly
nested under `phase_gates`; they were moved unchanged back to the top-level
`sources` list before adding this lane's source entry.

## BayesFilter usage audit

The new executable path and tests contain no import from MacroFinance-local
`filters.*`, `inference.hmc*`, `inference.mass_matrix`, or
`inference.posterior_adapter`; all active curvature, artifact, XLA, and HMC
types used here are BayesFilter-owned.

A broad MacroFinance repository scan still finds many pre-existing legacy,
performance, demo, and historical scripts that import the demoted local modules.
Those files were not run, edited, or used as evidence here. This result does not
claim repository-wide migration; any future active client run must pass its own
BayesFilter usage audit.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept common fixed-center curvature implementation | Passed: code/tests/docs/source map agree | No in-lane engineering veto remains | Real DZ5 target behavior | Run a separate reviewed foreign-binding exact-HMC mechanics canary | MAP, convergence, posterior validity |
| Keep both-binding dense projected geometry blocked | Existing MacroFinance raw curvature instability remains a veto | Veto supported by raw inertia/cross-fit evidence | Whether structured/consensus repair works on the real target | Apply this API to fresh both-binding clouds under a new experiment plan | Research-direction rejection |
| Offer CPU/XLA as explicit route | Exact spawn/import/XLA/lifecycle tests pass | Missing inherited env and child error fail closed | Scaling and workload-specific overhead | Benchmark only if CPU execution is explicitly requested later | CPU superiority or automatic fallback |
| Preserve legacy sequential score gate | Regression suite passes; fixed-center API is separate | No contract regression found | Global MAP remains unproved | Continue to use the legacy path only for its stated MAP-candidate semantics | Certified global MAP |

## Inference status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Engineering invariants pass; real both-binding dense geometry remains vetoed by prior raw instability |
| Statistically supported ranking | None; no stochastic candidate comparison was run |
| Descriptive-only differences | Prior DZ5 scores and curvature metrics remain descriptive/local evidence |
| Default readiness | Not established; caller-supplied stability caps remain required |
| Next evidence needed | Real-target fixed-center clouds, then an exact Metropolis-corrected mechanics canary, then separately planned retained multi-chain estimation and diagnostics |

## Run manifest

| Field | Value |
| --- | --- |
| Git start/checkpoint | `20835ecf90bff78ca93c5d401f231e4aa94e63ce` / `56b1f3c80831051148cd7bc1fd97d81a574f534c` |
| Final commit | selective result-bearing commit containing this note; hash reported in the execution handoff |
| Branch/remote | `main`; `origin` = `git@github.com:chakkeiwong/BayesFilter.git` |
| Environment | `tfgpu`; versions above |
| CPU/GPU | Explicit CPU diagnostic path; no GPU scientific or readiness run |
| Memory growth | Spawn children inherit and record `TF_FORCE_GPU_ALLOW_GROWTH=true`; GPU per-device verification N/A |
| Data | Deterministic/fixed-seed synthetic fixtures; MacroFinance artifacts are provenance only |
| Seeds | `20260716` family with disjoint `20260816` selection and `20260916` audit bases; other test seeds are fixed in source |
| Primary output | `docs/main.pdf`; this result note; source-map entry |
| Claude | Bounded read-only review, `VERDICT: REVISE`, all findings repaired |
| MathDevMCP | CLI doctor `ok: true`; four label/code checks `consistent` |

## Post-run red team

Strongest alternative explanation: synthetic linear score fixtures make local
curvature recovery much easier than the real DZ5 weak-direction problem. A
passing implementation suite could therefore coexist with an unusable real
geometry. That is why this result accepts engineering correctness but does not
promote the both-binding candidate or any default threshold.

Evidence that would overturn this result: a reproducible covariance/precision
orientation mismatch, audit-row use before selection, child bootstrap without
inherited CPU-only settings, non-SPD accepted artifact, or an in-scope test/math
failure. Evidence that would strengthen the next claim: fresh real-target
replicates satisfying predeclared raw/stability/holdout/audit caps followed by
an exact HMC mechanics canary with no divergences or numerical invalidity.

Weakest evidence: the CPU route is proven only as an exact child-process
engineering path, not as a performance choice; the stability thresholds are
caller hypotheses, not calibrated universal defaults.

## Remaining gaps to HMC estimation

1. Run the new API on fresh DZ5 foreign-binding and both-binding score clouds with reviewed target/data/code lineage and predeclared caps.
2. Keep the foreign-binding center labeled `diagnostic_center`; do not infer MAP attainment from stable curvature.
3. Run a separate exact Metropolis-corrected HMC mechanics canary using only an eligible artifact; record divergences, energy error, acceptance, movement, and target failures.
4. If mechanics pass, freeze the post-adaptation kernel under a retained-inference plan with exact seeds and raw sample retention.
5. Run multiple chains/replications and report R-hat, ESS/MCSE, divergences, prior and measurement-error sensitivity, and identification evidence before scientific interpretation.

No CPU scientific run, GPU run, HMC chain, or posterior estimation was executed
by this plan.
