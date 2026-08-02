# Projected-Cumulant GenUT Austria Rank Comparison Plan

Date: 2026-08-01  
Status: `SKEPTICALLY_AUDITED_READY_FOR_EXECUTION`

## Research Intent Ledger

| Field | Predeclared answer |
|---|---|
| Main question | Can a frozen low-rank subspace that matches complete projected third moments and fourth cumulants improve the Austria SIR finite-filter value stability and same-scalar score dispersion relative to diagonal and pairwise higher-moment corrections? |
| Candidate | The projected Tucker-cumulant correction derived in `docs/bayesfilter-genut-score-variance-problem-and-repair-note-2026-07-31.tex`, ranks `r=4,6,8` |
| Classification | `extension_or_invention`; Zhao--Cui motivates low-rank compression but does not claim this correction |
| Expected failure mode | Higher-moment residuals are particle noise rather than stable low-rank structure; the correction overfits calibration clouds or amplifies its own JVP |
| Promotion criterion | At `N=1008`, all claim rows are hard-valid; every score-coordinate SD is below both the diagonal and pairwise arm; paired aggregate score-variance uncertainty supports a reduction; value SD is no more than 25% above diagonal and the absolute mean-value shift is no more than one diagonal standard error |
| Promotion veto | Any non-finite value/score, invalid reset/OT, score-additivity failure, derivative-parity failure, mean/covariance restoration failure, normalized displacement above `2.0`, or value-stability gate failure |
| Continuation veto | Broken target/data identity, invalid basis artifact, ambient `d^3/d^4` runtime materialization, missing total-JVP term, GPU/XLA/memory-policy failure that cannot be repaired inside the budget, or all ranks failing mechanics validity |
| Repair trigger | One rank/control is invalid or descriptively worse; retain the artifact, select the least-displaced valid representative for that rank if available, and continue other ranks |
| Explanatory diagnostics | Heldout higher-moment subspace energy, principal angles, projected tensor residuals, finite-time tangent growth, SGQF distance, runtime, and allocator peak |
| Must not be concluded | No exact Austria likelihood/score, score-bias reduction, source-faithful Zhao--Cui result, universal rank law, HMC readiness, default change, or statistical superiority from the three-seed `N=4032` diagnostic |

## Claimed And Computed Quantities

The claimed runtime score is the total derivative of the same executed finite
particle-filter scalar with a frozen projected-cumulant correction. The
implementation must include source particles and weights, whitening,
projected target tensors, current-cloud tensors, residual contraction,
affine-tangent projection, direction normalization, restandardization, and
source mean/covariance restoration.

Austria has no exact nonlinear score oracle in this campaign. Therefore:

- score SD is particle-seed dispersion, not absolute score error;
- `N=1008 -> N=4032` displacement is a finite-particle stability diagnostic;
- SGQF value/score is an existing comparator, not an oracle;
- value shifts between finite programs are descriptive unless a valid external
  reference is supplied.

## Baseline Ladder

| Arm | Role |
|---|---|
| No higher-moment correction | Naive mechanics baseline, if affordable after required arms |
| Diagonal correction | Current tuned classical baseline: four steps, strength `0.2` |
| Pairwise correction | Previous proposed repair: diagonal controls plus four pairwise steps, strength `0.02` |
| Projected `r=4` | Smallest proposed full projected-tensor arm |
| Projected `r=6` | Intermediate proposed arm |
| Projected `r=8` | Largest bounded proposed arm and calibration parent basis |
| Existing fixed SGQF | External descriptive comparator: value `-682.3480055392419`, score `[28.739453057371584, -106.65885657030441, 9.43117639262833]` |

The no-correction arm is secondary and may be omitted if the bounded GPU
budget is consumed by required ranks. Diagonal, pairwise, and all three rank
arms are mandatory unless a hard continuation veto fires.

## Exact Scope

- Model/target: `austria_sir_T20`, transition before observations `y1..y20`.
- State/observation/parameter dimensions: `18/9/3`.
- Particle counts: `N=1008` and legal `N=4032`; `N=4000` is invalid because
  the exact replicated cubature design requires divisibility by `36`.
- Claim seeds:
  - `N=1008`: `98201..98216`;
  - `N=4032`: `98201..98203`, descriptive capacity/stability only.
- Calibration observations: target-owned seeds `91141,91142`.
- Validation observations: target-owned seeds `91241,91242`.
- Tuning particle seeds: `98301,98302`.
- Backend: TensorFlow 2.19.1, GPU/XLA, `float32`, TF32 enabled to compare the
  existing route at its active policy, verified memory growth before logical
  device initialization.
- Versioned output root:
  `docs/benchmarks/artifacts/projected_cumulant_genut_austria_20260801/`.

Every `(N,r)` pair is a distinct tuning scope. Settings and bases may not be
transferred across particle counts. Within one particle count, ranks 4 and 6
are nested prefixes of the frozen calibration rank-8 basis so rank comparisons
do not change the learned parent subspace.

## Mathematical And Source Audit

The mathematical algorithm, total-JVP formulas, cost, and nonclaims are in
Section `sec:gsv-projected-cumulants` of the LaTeX note.

Inspected primary/source anchors:

- local paper: `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf` and `.txt`, equations (12)--(16), Algorithms 1--2, Proposition 2, and the complexity discussion;
- author sequential route: `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21` and `:46`;
- author squared-TT marginal contraction: `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25`.

Allowed source claim: Zhao and Cui use low-rank TT/squared-TT density
approximations and marginal contractions. Forbidden source claim: Zhao and Cui
propose projected cumulant matching. The latter is a project derivation.
Network metadata, forward snowballing, and venue/citation counts are not needed
to classify this implementation and are not claimed.

## Offline Basis Protocol

For each `N`, run the diagonal baseline on the two calibration datasets and
two tuning particle seeds. At every time step, whiten the weighted source and
uniform reset clouds and form deterministic mode-one residual sketches of the
complete third moment and fourth cumulant. Use fixed unit sketch directions
with a recorded stateless seed. Normalize order-three and order-four sketches
by declared Gaussian noise scales before accumulating the positive
semidefinite mode score `H_t`.

Compute a deterministic descending eigendecomposition of each `H_t`; canonicalize
each eigenvector sign by making its largest-magnitude entry positive. Freeze
the first eight columns as `U_t`. Rank 4 and 6 use prefixes.

On the untouched validation datasets:

- report explained validation mode-score energy for `r=4,6,8`;
- fit a diagnostic validation basis and report principal angles against the
  calibration basis;
- never replace the calibration basis using validation results;
- treat weak stability as a promotion veto/explanation, not a continuation
  veto, so the requested downstream rank comparison is still executed.

## Rank-Specific Control Tuning

Keep diagonal controls fixed. For every `(N,r)`, tune only projected controls
on validation data:

```text
projected_steps x projected_strength =
  (1, 0.0025), (1, 0.005), (1, 0.01),
  (2, 0.0025), (2, 0.005), (2, 0.01)
projected_floor = 1e-5
```

Selection order:

1. hard numerical, mean/covariance, score-additivity, and displacement gates;
2. no score-coordinate variance above the diagonal validation baseline;
3. value SD no more than 25% above diagonal and mean shift within one
   diagonal validation standard error;
4. lowest maximum score-coordinate variance ratio;
5. lowest aggregate score variance, projected residual, displacement,
   strength, then step count.

Numerical residuals use the existing Austria GenUT lane tolerance `5e-4`, as
declared by the July 23 whole-leaderboard and higher-moment retuning harnesses;
the normalized displacement veto is `2.0`. Validation particle variance is
computed separately inside each synthetic observation dataset and then averaged
across datasets. Observation-to-observation variation is not pooled into the
particle-seed variance used for tuning.

If no nonzero control passes every validation veto but at least one is valid,
freeze the least-displaced valid representative and label it
`INELIGIBLE_DIAGNOSTIC_REPRESENTATIVE`; run it on claim data to answer the
rank-comparison question without calling it promotable. Claim data is never
used for tuning.

## Implementation Phases

1. Add projected full third/fourth tensor contractions and a complete manual
   JVP to `higher_moment_contract_e.py`; never form ambient dense tensors.
2. Wire frozen per-time bases and projected controls through
   `cubature_genut_filter.py`, with diagnostics and zero-control parity.
3. Add diagnostic higher-moment mode-score capture for offline basis fitting.
4. Add focused tests: exact projected tensors, direction versus autodiff,
   manual JVP versus autodiff/centered finite differences, affine restoration,
   zero-control parity, nested-basis determinism, and XLA graph inspection.
5. Add the bounded Austria campaign runner with complete manifest and fresh
   artifacts.
6. Run CPU-hidden compile/focused tests and LaTeX build.
7. Run one trusted GPU/XLA smoke.
8. Fit and validate bases, tune ranks, and execute untouched `N=1008` claims.
9. Execute the `N=4032` three-seed diagnostic with sequential processing if
   required for memory.
10. Write terminal result and post-run red-team notes.

## Evidence Contract

| Role | Diagnostic |
|---|---|
| Hard veto evidence | Non-finite value/score, invalid program/reset, score-additivity failure, mean/covariance violation, derivative parity failure, target mismatch, missing artifact |
| Promotion criterion | The complete `N=1008` rank-vs-baseline value/score gates stated above |
| Promotion veto | Weak/unstable heldout subspace, value-stability failure, or tangent-growth deterioration even if score SD is lower |
| Continuation veto | Invalid implementation/target/artifact or no mechanically valid rank |
| Repair trigger | Rank/control-specific failure under an otherwise valid harness |
| Explanatory only | SGQF gaps, extreme growth summaries, runtime, allocator peak, and `N=4032` three-seed differences |
| Terminal artifacts | JSON, Markdown, basis/tuning artifacts, run manifest, and result note under the versioned root |

## Skeptical Plan Audit

| Risk sought | Audit finding and repair |
|---|---|
| Wrong baseline | Use both current diagonal and pairwise routes with identical data/noise; SGQF remains a comparator, not truth |
| Proxy promoted to criterion | Explained subspace energy and residual loss are diagnostics/vetoes only; downstream value/score is primary |
| Missing stop conditions | Explicit hard continuation vetoes, per-rank repair triggers, and bounded attempts are declared |
| Unfair rank comparison | One nested rank-8 calibration basis per `N`; rank prefixes and common claim seeds |
| Hidden assumption | Low-rank higher cumulants are a hypothesis, tested by heldout energy and principal angles before interpretation |
| Stale controls | Projected controls tune separately for every `(N,r)`; inherited diagonal controls remain a frozen baseline rather than a projected default |
| Covariance PCA misuse | Rejected because whitening makes covariance identity; basis is fitted from higher-moment residual sketches |
| Claim leakage | Basis uses calibration data, control selection uses validation data, and claim observations/seeds remain untouched |
| Discontinuous derivative | Basis/rank are frozen; no eigendecomposition, sign selection, or rank adaptation occurs in the claim graph |
| Partial derivative | Total-JVP checklist and parity tests cover targets, weights, both clouds, projection, normalization, and restoration |
| Misleading successful command | Artifacts must include downstream value/score and hard validity, not only tensor residuals |
| Underpowered ranking | Sixteen common seeds at `N=1008`; `N=4032` three-seed output is explicitly descriptive only |
| Compute explosion | Maximum six controls per `(N,r)`, three ranks, two particle counts, one smoke, and fresh attempt roots |
| Environment mismatch | Trusted/elevated GPU, XLA, TF32, and verified memory growth are required and recorded |

Audit verdict: `PASS_AFTER_REVISION`. Material flaws in the initial idea were
repaired before execution: covariance PCA was replaced by cumulant-residual
sketches; claim-data basis fitting was prohibited; downstream metrics replaced
residual loss as the promotion criterion; and the larger-N comparison was
downgraded from ranking evidence to a descriptive stability diagnostic.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Ranks `4,6,8` | User request | Candidate hypotheses | Too small misses structure; too large fits noise | Heldout energy, angles, downstream ladder |
| Mode-score sketch | Project derivation | Hypothesis | Sketch directions miss important modes | Fixed oversampled sketches and validation energy |
| Nested rank-8 parent | Fair-comparison requirement | Reviewed convenience | Leading rank-8 space unstable | Per-time principal angles |
| Six-control grid | Bounded local tuning ladder | Convenience grid | Optimum lies outside grid | Boundary-selection flag; no expansion after claims |
| Existing diagonal controls | July 23 Austria scope | Frozen baseline | Not optimal after new correction | Rank-specific validation tuning; no default claim |
| Numerical residual tolerance `5e-4` | Existing Austria GenUT claim harness | Frozen baseline gate | A looser tolerance could admit invalid arithmetic; a stricter uncalibrated tolerance could reject float32 additivity roundoff | Persist each residual and validity check per row |
| Within-dataset validation variance | July 30 pairwise tuning protocol | Reviewed estimator | Pooling datasets confounds observation variability with particle MC variance | Persist dataset identity and per-row seed |
| Pairwise controls | July 30 selected arm | Historical comparator | Pairwise instability at larger N | Reproduce hard validity and score SD |
| TF32 | Active production direction and prior route | Scope setting | Numerical score drift | Same-route parity tests and hard finite gates; no cross-dtype claim |
| SGQF comparator | Existing observed-data artifact | Descriptive comparator | Different approximation bias | Never used as an exact oracle or tuning target |

## Budget And Attempt Policy

- CPU/reference phase: one focused suite plus at most two localized repairs.
- GPU smoke: one attempt plus at most two infrastructure/mechanics repairs.
- Basis phase: one calibration and one validation pass per particle count.
- Tuning: at most six projected controls for each of six `(N,r)` scopes.
- Claim: one `N=1008` and one `N=4032` pass per frozen arm.
- Expected bounded wall time: up to three GPU hours; stop before expansion.
- Every retry uses a new versioned output directory and records the failure,
  repair, wall time, and remaining budget.

## Post-Run Reporting Requirements

The terminal note must include:

- per-arm value/score means, SDs, paired uncertainty, and hard vetoes;
- whether any rank is statistically supported over both baselines;
- descriptive-only `N=4032` differences and N-displacement;
- heldout basis stability and residual-energy diagnostics;
- runtime and GPU memory;
- decision and inference-status tables;
- strongest alternative explanation, weakest evidence, overturning evidence;
- explicit candidate rejection versus research-direction rejection; and
- what is not concluded.

## Execution Ledger

| Attempt | Outcome | Classification and repair |
|---|---|---|
| CPU/reference and document gate | `17 passed`; LaTeX built to 15 pages; static compile and `git diff --check` passed | Mechanics gate passed |
| `N=1008` smoke attempt 01 | XLA compiled on RTX 4080 SUPER, then basis capture raised an opaque validity exception | Harness defect: an undocumented `2e-4` residual tolerance replaced the established Austria `5e-4` gate, and failure rows were not persisted |
| Smoke repair | Restored the established `5e-4` tolerance, changed tuning variance to within-dataset particle variance, added per-row validity checks and failure artifacts | Localized harness repair; target, method, data partitions, criteria, hardware, and campaign budget unchanged |
| `N=1008` smoke attempt 02 | Completed in 135.84 s; diagonal, pairwise, and `r=6` valid; first-grid `r=4` and `r=8` invalid; maximum heldout principal angles `89.65`, `89.73`, and `88.38` degrees for ranks 4, 6, and 8 | Rank-specific tuning repair trigger, not a continuation veto. Weak basis stability is already a promotion veto; downstream comparisons remain required to determine candidate versus direction rejection |
| `N=1008` claim attempt 01 | Basis, all rank tuning, and all claim arms completed; terminal bootstrap reporting raised `IndexError` because invalid ranks have incomplete seed sets | Harness repair: align bootstrap rows by common seed and return `available=false` for incomplete arms. Checkpoint preserves the completed scientific work; rerun uses a fresh output root |
| `N=1008` claim attempt 02 | Completed in 316.64 s. Ranks 4 and 8 had no mechanically valid tuned control. Rank 6 selected an ineligible diagnostic representative, failed 2 of 16 claim seeds, and therefore has no admissible variance comparison. Diagonal and pairwise were fully valid; their score SDs were respectively `[3435.63, 1272.44, 301.97]` and `[33.94, 17.99, 19.72]` | Projected candidate rejected at `N=1008`. Pairwise remains viable in this scope. Continue the separately tuned `N=4032` descriptive phase because it tests the predeclared particle-count dependence and no continuation veto fired |

The active runner checkpoints after basis validation, each rank-specific tuning
step, and each claim arm. A rank with no mechanically valid tuned control is
recorded and skipped while viable ranks continue; the campaign stops only if
all ranks fail mechanics tuning, as declared above.
