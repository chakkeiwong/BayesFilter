# Zhao-Cui/GenUT Dual-Cap Cross-Model Test Plan

Date: 2026-08-07

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | Does the Austria dual-cap numerical map remain finite, internally differentiable, and variance-competitive when applied to the existing LGSSM, KSC-SV, and predator-prey GenUT candidate routes? |
| Candidate | Existing diagonal GenUT correction plus pairwise correction, optional smooth radial pairwise direction cap, and final smooth coordinatewise cap `f_b(x)=x/(1+(x/b)^8)^(1/8)` with `b=0.98`. |
| Baseline | The same finite GenUT program, same observations, particle noises, design, transport controls, and claim seeds, with all moment correction and cap controls disabled. A second comparison arm keeps the previously selected diagonal-only moment correction but disables pairwise and both caps. |
| Primary promotion criterion | Candidate and baseline all pass finite/program, transport residual, score-increment, and derivative finite-difference gates on every claim row; candidate is a viable numerical arm, not a superiority claim. The final report must also state when the baseline itself fails the derivative gate. |
| Promotion veto | Any non-finite output, invalid program/chart, GPU/XLA mismatch, transport residual over the existing `5e-4` gate, or failed same-scalar FD gate. |
| Continuation veto | Missing target hash, missing baseline control provenance, corrupted artifact, or inability to execute a model's declared route. A model-specific candidate failure is not a veto on the other models. |
| Explanatory diagnostics | Cap activity/displacement, inverse derivative, shape residuals, paired value/score shifts, per-coordinate seed SD, and exact/dense reference error where available. |
| Nonclaims | No claim of Zhao-Cui source faithfulness, exact nonlinear likelihood/score, posterior correctness, HMC/NeuTra readiness, default promotion, or broad superiority. |

## Scope And Model Semantics

| Model | Horizon | State/parameter dimensions | Reference role | Pairwise status |
|---|---:|---:|---|---|
| LGSSM | 50 | 3/5 | Exact Kalman value and score | Active; six ordered off-diagonal moment entries |
| KSC transformed SV mixture | 10 | 1/2 | Dense transformed-mixture value/FD score diagnostic | Structural no-op at state dimension one |
| Predator-prey | 20 | 2/6 | No exact nonlinear authority; prior fixed-variant rows are diagnostic comparators | Active; two ordered pairs |

Austria SIR is not rerun here because its strict bounded teacher/cap campaign is
already recorded at `T=2`; the cross-model route tests the shared candidate map
on the three remaining model families. The coordinatewise cap is treated as an
extension/invention outside the Austria bounded-teacher chart.

## Controls

Transport controls are inherited from each model's existing scope-specific
GenUT tuning artifact and are not retuned on claim rows. For each model the
candidate grid is:

1. diagonal-only moment correction, no pairwise correction, no caps;
2. pairwise correction with the selected prior pairwise controls, no caps;
3. pairwise correction plus radial cap `2.0`, no coordinate cap;
4. pairwise correction plus coordinate cap `b=0.98`, `p=8`;
5. pairwise correction plus radial cap `2.0` and coordinate cap `b=0.98`, `p=8`.

The no-correction arm is the common particle baseline. Candidate selection is
calibration-only and chooses the least-displacing arm among those that pass all
calibration validity gates. KSC's pairwise arms are retained as an explicit
structural-null diagnostic and must be identical to their corresponding
diagonal/cap arm up to floating-point evaluation.

## Evidence Contract

- Same target observations, event order, parameter point, design, particle
  count (`N=1008`), dtype (`float32`), TF32 enabled, GPU/XLA, and seeds are used
  for every arm within a model.
- Calibration uses the existing two disjoint tuning seeds; validation uses six
  untouched common particle seeds (`98201`--`98206`) and paired rows.
- Every row records finite/program validity, device, transport residuals,
  score-increment sum residual, cap diagnostics, and value/score.
- Every selected arm receives a same-scalar centered FD check using common
  noises on all parameters at `h=1e-3`; absolute `<=0.08` and normalized
  `<=0.03` are hard gates.
- LGSSM absolute error to Kalman is a primary explanatory accuracy diagnostic;
  KSC dense-reference error is diagnostic only; predator-prey has no accuracy
  authority.
- The artifact is written below a fresh versioned output root with a command,
  git commit, environment, device, memory policy, seeds, source hashes, and
  plan path.

## Default/Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| `b=0.98,p=8` | Austria T2 calibration | Least-displacing valid Austria arm | Model chart scale may make it active too often | Per-row displacement and active fraction | Hypothesis, not transferred default |
| Existing per-model transport controls | Scope-specific prior tuning artifacts | Preserves target and computation budget | Stale controls could confound comparison | Hash and scope checks before execution | Baseline inheritance |
| `N=1008`, six seeds | Existing whole-leaderboard contract | Bounded cross-model cost | Few seeds cannot establish ranking | Paired SD/MCSE and descriptive-only language | Feasibility evidence |
| TF32 GPU/XLA | repository production target | Tests intended execution backend | Precision may alter small shifts | Device/TF32 manifest and FD gate | Candidate backend |
| Coordinate cap outside Austria teacher | BayesFilter extension | Tests shared numerical map | Changes moments/objective and may hide bias | Cap activity, paired shifts, references where available | Extension/invention |

The first cross-model retry showed that a generic cap implementation must retain
the existing weighted source mean/covariance after capping. That repair was
tested by primitive affine-restoration and JVP tests before the claim retry.
The final retry also showed that the inherited FP32/TF32 baseline itself can
fail the `h=1e-3` FD diagnostic on LGSSM and predator-prey; this is an
implementation/precision diagnostic, not evidence that the cap caused those
failures.

## Pre-Mortem And Stop Conditions

The run could pass while misleading us if the cap changes the scalar materially,
if inherited controls are stale, or if the reference is treated as an oracle
for a different target. These are addressed by paired baseline rows, scope/hash
checks, cap displacement diagnostics, and explicit reference roles. It could
fail from an implementation/backend issue rather than the idea; a failed model
gets a bounded CPU replay and source inspection before being classified.

Stop the campaign when all three models have completed the declared rows, or
when the total budget reaches 90 GPU minutes. Do not retry with changed targets,
seeds, particle counts, controls, or cap definitions under this plan.

## Planned Artifacts

- Runner: `docs/benchmarks/run_zhao_cui_genut_dual_cap_cross_model.py`
- JSON/Markdown result root:
  `docs/benchmarks/artifacts/zhao_cui_genut_dual_cap_cross_model_20260807/`
- Result note:
  `docs/plans/bayesfilter-zhao-cui-genut-dual-cap-cross-model-result-2026-08-07.md`

## Skeptical Plan Audit

The plan passes audit with the following limitations recorded rather than
hidden: the selected cap is transferred only as a candidate hypothesis; KSC
pairwise correction is structurally void; LGSSM and KSC references do not
validate the predator-prey nonlinear target; and few-seed variance differences
remain descriptive. The artifacts answer numerical viability and internal
derivative consistency, not exact score accuracy or method superiority.
