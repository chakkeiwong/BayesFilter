# Phase 4B matrix-LGSSM campaign amendment

Date: 2026-09-09
Status: implementation and preflight authorized; campaign follows passed gates

This amendment executes Step 6 of the 2026-09-08 resampling-reference plan.
The 2026-09-09 continuation note records branch reconciliation and the repaired
GPU/XLA evidence. No scientific target, method family, or promotion threshold
is expanded here.

## Research intent and evidence contract

Question: does the complete raw-IWSG mixture resampler, after the canonical
Contract-E/GenUT dual-cap reset, estimate the exact LGSSM score with lower
mean squared error than the canonical finite score?

Candidate: `RESKDM-IWSG-FINITE`, using every mixture component, fixed-anchor
sample/proposal banks, and total analytical derivatives through raw weights,
covariance marks, flow, PF-PF factors, and reset. Both existing mark policies
enter calibration. One parameter controls the full two-state transition, so
the one-direction analytical endpoint supplies its entire score.

Comparators are the public `canonical_value_and_analytical_score` endpoint
(`ATOM-FINITE`), independently implemented matrix Kalman value/score, the
fixed-stream bootstrap PF, and the freshly calibrated Phase 4A observation-KDM
route. All finite methods see the same observations, initial particles, and
compatible process streams. The resampling streams are paired where their
meaning agrees. The auxiliary control variate is absent because a suitable
zero-mean construction has not been proved. An additional innovation-jitter
arm belongs to the broader master program and is not silently claimed here.

Promotion requires a paired 95% interval demonstrating at least 10% MSE
reduction against the canonical method on untouched validation paths. Compute
both the interval for `e_B^2-e_A^2` and the interval for
`e_B^2-0.9*e_A^2`; the latter must have its upper endpoint below zero. This
accounts for uncertainty in the canonical MSE itself. A merely negative point
estimate cannot promote.

Hard continuation vetoes are wrong target/route identity, nonfinite or invalid
results, failed derivative or replay checks, a missing pairwise correction,
failed canonical/zero-bandwidth parity, corrupted evidence, or exhausted total
budget. Failed candidate MSE is a repair trigger, not a continuation veto.
Runtime, ESS, mixture entropy, cap activity, bandwidth, and point rankings are
explanatory. No outcome proves equality with `ATOM-FINITE`, unbiased model
scores, DSGE support, HMC validity, or production/default readiness.

## Fixed model, numerical scope, and assumptions

Use exactly the matrices and transition-first timing in the parent plan and
LaTeX equation `phase4b-matrix-lgssm`: `theta=0.72`,
`A=[[0,0.12],[-0.08,0.10]]+theta*diag(1,0.7)`,
`H=[[1,0.25],[-0.15,0.9]]`, `Q=[[0.12,0.025],[0.025,0.09]]`,
`R=[[0.22,0.035],[0.035,0.18]]`, `P0=[[0.8,0.12],[0.12,0.6]]`,
and initial mean zero. The first observation follows one transition of a draw
from `N(0,P0)`. Every particle starts with local covariance mark `P0`.

Scopes are `(N,T)=(32,5)` followed by `(64,20)`. The default campaign dtype is
float64, TF32 off, XLA on, on the RTX 4080 SUPER. This is the numerical
reference campaign, not a production throughput test. Float32/no-TF32 GPU
smokes provide separate implementation evidence; precision comparisons must
not confuse different stateless draws with arithmetic error.

| Choice | Provenance and justification | Failure mode / earliest check | Status |
|---|---|---|---|
| Full two-state matrices | Parent plan; correlated Q/R and non-diagonal H exercise matrix recurrences | Wrong timing or model: exact oracle FD and matrix spectral radius | Frozen model |
| Full reset and both corrections | Owner's algorithm definition | Vacuous correction: nonzero off-diagonal mask and pairwise displacement | Required |
| Six flow substeps; 4 Sinkhorn and 2 balance iterations | Small Phase 4A numerical baseline, not promoted settings | Inadequate finite approximation: oracle error reported per scope; cannot promote the baseline | Diagnostic baseline |
| Explicit ridge `1e-5`, diagonal strength `0.2`, LM damping `1e-2`, LM scale floor `1e-4`, trust radius `0.5`, pairwise strength `0.02`, RMS cap `2`, coordinate cap `0.95`, power `8` | Shared finite-program baseline; retain the owner's capped route while examining KDM | These define the finite scalar and may dominate model error; paired baseline, cap diagnostics, and FD establish only mechanics | Frozen hypotheses; not tuned defaults |
| Residual design `(+e1,+e2,-e1,-e2)` tiled to N | Full-rank, zero-mean, balanced deterministic design for Contract-E | Degenerate design: rank check and finite reset/cap trace | Declared design |
| One annealing stage | Supported Phase 4B scope | Could be an inferior baseline on harder data; no cross-model promotion | Scope restriction |
| `B=rho^2 Q`, eight positive rhos from parent | Tests scale while preserving process covariance geometry | Bandwidth bias/high IWSG variance: calibration and fresh validation | Hypothesis |
| Mean or selected covariance mark | Explicit local extensions | Downstream UKF sensitivity: calibration ablation and full FD | Hypotheses |
| Float64/no TF32 | Passed correctness smokes; avoids known TF32 admission failure | Reference cost may exhaust budget: compile and warm timing before calibration | Reference authority |
| Fixed numerical controls across compared arms | Defines comparison of KDM insertion into the same finite baseline | Results conditional on these controls; scope-specific tuning still owed before canonical promotion | Diagnostic comparison only |

These frozen controls have not acquired default status. This campaign can
reject or nominate KDM in the declared finite baseline; broad promotion still
requires target-specific numerical calibration under the owner tuning policy.

## Data, calibration, and power

Generate all normal variates as standard normals (no shrinkage of KDM noise),
stratified offsets on `[0,1)`, and observations from the stated generative
model. Use TensorFlow stateless streams. A stream key encodes the attempt,
scope, split, replicate, and tensor role without arithmetic collisions.
Preflight, calibration, power pilot, and validation are disjoint. No old
Phase 4A holdout or setting is reused as selection evidence.

Calibration uses 40 independent paths per scope for each of
`rho=(0.025,0.05,0.1,0.2,0.4,0.8,1.2,1.6)` and each mark policy. Select the
lowest calibration MSE for Phase 4B, breaking ties by grid order. Select the
Phase 4A rho separately. Selection is nomination, not statistical ranking.
Freeze both choices and the calibration median of absolute exact scores
before the 30-path power pilot. This median defines low/high-score groups.

Power must match the promotion boundary. Let
`D=e_B^2-0.9*e_A^2`. Under the design alternative of a 20% true MSE reduction,
the distance from the required 10% boundary is
`delta=0.10*canonical_calibration_MSE`. Use the pilot sample SD of D in
`n=ceil((1.96+0.84)^2*SD(D)^2/delta^2)`, round upward to a multiple of 20,
and bound executed n to 100--500. This is nominal 80% power for that *declared
20% alternative*, not a promise about the unknown true effect or the
probability of passing all vetoes. This repairs the original formula's
unstated relationship to the ten-percent promotion boundary.

If required n exceeds 500, execute at most the 500-path descriptive holdout
only when the remaining budget permits, label it underpowered, and prohibit
promotion. Preserve required and actual counts. Bootstrap 4,000 paired path
resamples for intervals. Report bias, sample variance, MSE, error-mean MCSE,
paired difference MCSE, and intervals. Bootstrap/reporting is a host-side
TensorFlow diagnostic, outside GPU kernel timing.

## Heuristic dominance and uncertainty

Constructed cheap adversaries: the canonical unmodified filter tests whether
adding KDM is useful; the bootstrap filter tests whether flow/reset complexity
helps; the observation-convolution route tests whether continuous resampling
adds information beyond a simpler KDM insertion. The exact Kalman score is
the certifying oracle. The absent control variate is recorded explicitly.

For each `(N,T)` scope and both frozen low/high-score groups, report each
implemented comparator's MSE and paired intervals against Phase 4B. Any lower
comparator point MSE is a conservative promotion veto as in the parent plan;
call it a statistically supported loss only when its interval excludes zero.
Fewer than 30 validation paths in either group blocks promotion for inadequate
conditional evidence. These groups are not tuning targets.

## Runner, budget, stop conditions, and retained output

Implement `docs/benchmarks/run_ledh_younis_kdm_phase4b_campaign.py` with stable
TensorFlow signatures and the existing numerical endpoints. It must expose
`--mode preflight` and `--mode campaign`; both default to GPU/XLA. Small CPU
test fixtures explicitly hide CUDA and are diagnostic exceptions.

Preflight uses N=8,T=2 and checks the actual matrix fixture, exact oracle FD,
canonical and Phase 4A-zero parity, complete fixed-anchor replay FD for both
mark policies, raw-weight recurrence, all-pairs count, full cap diagnostics,
bootstrap realized-index FD, and host-consumed validity flags. A bootstrap FD
comparison is valid only if its ancestor indices stay fixed across the
perturbations. Implementation preflight is not a score-quality comparison.

Campaign command, from the repository root:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_ledh_younis_kdm_phase4b_campaign.py --mode campaign --output-dir docs/benchmarks/artifacts/ledh_younis_kdm_phase4b_20260909/campaign01 --attempt 1 --budget-seconds 2700
```

Campaign budget: 2,700 seconds total including compilation and all paths; at
most two localized implementation retries within the same remaining budget.
Check elapsed time before each path/candidate evaluation. After warm timing,
project the cost of the remaining calibration and minimum validation. Stop
as under-budgeted before validation if it cannot fit; do not downsample the
prespecified test and call it powered. The same check controls the second
scope. A scientific or numerical veto preserves the failed row and stops
that attempt. A failed but valid score comparison continues to the next
scope if affordable.

Every attempt creates a new directory and writes its manifest first, progress
and path rows incrementally, and a terminal result even on a caught failure.
Record Git commit, exact command and environment, source hashes, model,
seeds/stream scheme, memory-growth verification, dtype/TF32/XLA, timing,
allocator peak, selected settings, required/actual power counts, and verdicts.
Use JSON without nonstandard NaN tokens. Compact manifests/results are
retained in Git; raw path rows and progress remain ignored.

## Skeptical audit before implementation

The plan passes with two explicit repairs: the power calculation now names an
alternative separated from the actual promotion threshold, and the document
must correct its zero-bandwidth sampling statement for uniform stratified
weights. Wrong baseline, scalar no-op correction, proxy promotion, silent
defaults, sample reuse, missing covariance terms, and XLA assertions are
covered by the endpoint preflight and explicit host validity checks.
Numerical controls remain unpromoted hypotheses, so this result cannot claim
the best tuned classical baseline or close the broader promotion ladder.

The most serious remaining risk is under-budgeting the large scope or the
validation count needed for heavy-tailed IWSG errors. Timing projection and the
power cap must report that directly. A finite candidate failing MSE tests
rejects the candidate in this scope; it does not reject the entire KDM idea.
