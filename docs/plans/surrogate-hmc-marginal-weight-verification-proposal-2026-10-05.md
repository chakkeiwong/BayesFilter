# Proposed verification of the marginal-weight improvement

Status: proposed follow-up; not executed. The original marginal-weight master
program is complete. This extends that program's evidence rather than changing
its completed verdict. The overall goal remains **testing surrogate HMC**;
the first stages below test its LEDH likelihood and analytical score component.

## Question and current evidence

Does replacing ancestor weights with marginal-mixture weights reduce error
across independent data, parameter regimes and particle designs, while
preserving the LGSSM and KSC calculations? The completed experiment supports
material non-inferiority in LGSSM/KSC and lower error in the four tested M13
cells. It does not establish improvement in every model. References, detailed
uncertainty and remaining failures are in
[the result note](surrogate-hmc-marginal-weight-results-2026-10-05.md).

The comparator is the same canonical LEDH implementation, with only
`importance_weight_policy` changed. Use common observations and paired particle
designs, identical scope-specific controls for the weight-law ablation, and
record any separately tuned comparison as a different experiment. Retain
FP32/TF32/XLA as the candidate mode; FP64 checks are independent references.

## Verification ladder

1. **Close engineering gaps cheaply.** Exercise the real batch consumer with
   multiple distinct parameter rows, checking each against the single-row
   result for both policies. Check every parameter direction against finite
   differences on fresh designs. Repeat N=4096 M13 evaluations and compare
   FP32 with FP64 at the same rounded inputs. The existing N=256 precision
   check does not cover the N=4096 results. Distinguish score sensitivity from
   changes in the endpoint likelihood; preserve all validity diagnostics.
2. **Replicate on fresh datasets.** Keep the existing LG1 T=2, LG3 T=2/10,
   KSC T=10/20 and M13 T=1/20 scopes and their declared parameter points.
   Start with four new datasets per scope and 32 paired designs per dataset;
   extend cumulatively to eight and then sixteen datasets if uncertainty is
   unresolved and the budget permits. Use new calibration/validation data to
   freeze controls before generating claim results. Existing claim datasets
   are historical comparisons, not new holdouts. Do not retune on a failed
   claim dataset. Report every model/regime separately.
3. **Strengthen the accuracy authorities.** Use exact Kalman likelihoods and
   scores for LGSSM; independently refined integration for the same KSC
   seven-component observation model; and refined quadrature for M13 T=1.
   For M13 T=20, compare replicated bootstrap/Fisher references at N=65536,
   131072 and, if needed, 262144. Refinement agreement in MCSE units is a
   diagnostic, not proof of zero particle bias. Check sensitivity of the
   comparison to both reference uncertainty and observed refinement shifts;
   unresolved shifts make the affected accuracy verdict inconclusive.
4. **Measure cost and residual error.** After fresh scope-specific tuning,
   use N=1024/4096/16384 to test sensitivity to particle count. Include prior
   importance sampling, bootstrap filtering, and Gaussian filtering (Kalman,
   collapsed-mixture KSC, and M13 EKF), with their model-specific limitations
   stated. Compare accuracy at equal particle count and separately at equal
   measured wall time, including compilation/tuning cost where relevant.
   Numerical convergence with N is explanatory: deterministic reset and
   filtering approximation errors need not disappear as N increases.

Stages 1–3 are the first proposed execution block. Stage 4 follows only within
its remaining budget; otherwise it needs a new resource estimate. Before
launch, implement dataset/seed controls and the statistical aggregation in the
existing master, estimate cost on a disjoint calibration dataset, and record
exact commands and a fresh output root. Proposed cap: six GPU-hours total,
including references, scope tuning and repairs; one bounded infrastructure
retry per phase. This is a proposed allocation, not a claim about previously
granted deadlines or an assertion that every rung will fit.

## Decision rules and assumptions

For each dataset and design, compute the candidate-minus-ancestor difference
in absolute log-likelihood error and in the existing physical-score RMS error
scaled by 0.1. Aggregate paired designs within datasets before assessing
cross-dataset uncertainty; designs sharing a dataset are not independent
datasets. Predeclare simultaneous one-sided intervals across all cells, both
metrics and the three possible looks. Use dataset-clustered uncertainty and
report hierarchical-bootstrap sensitivity; with few datasets, interval
approximations remain a limitation. Do not report an unadjusted interval after
repeatedly checking for success.

The inherited 0.05 margins support the same local material non-inferiority
claim when every required upper bound is below 0.05. An upper bound below
zero supports lower error in that cell; it does not automatically establish a
practically important gain. Report magnitudes as well as intervals. Claim
improvement in all three models only if each model's required conditional
comparisons support that stronger statement. LGSSM already has an exact
Kalman authority, so non-deterioration can be the appropriate outcome.

The 0.1 score scaling and 0.05 margins come from the completed plan and are
local comparison conventions, not established HMC tolerances. The particle
ladder tests resource sensitivity; it is not a proven convergence rate. The
dataset counts and 32 designs are initial budget hypotheses: use the first
disjoint calibration block to forecast interval width and cost, and report
underpowered results as inconclusive at the cap. Fixed shared controls isolate
the weight law; they do not prove either policy optimally tuned.

Nonfinite values, invalid covariance and wrong total derivatives veto the
affected implementation. Broken reference accuracy, corrupted data or invalid
pairing stop the affected comparison for repair. A candidate that loses to a
simple comparator fails promotion but may continue to the planned diagnosis.
Save raw per-dataset pairs, reference refinements, source/seed/environment
manifests, elapsed budget and the refreshed next-phase decision. The strongest
alternative explanation for the present gain is unusually favorable data or
particle designs; independent datasets address it directly. Stronger M13
references address a separate uncertainty about its measured absolute error.

## Downstream surrogate-HMC test

Component verification is followed by a separately budgeted surrogate-HMC
comparison against model-specific posterior references. Define the endpoint
target and proposal force separately, freeze each scope's tuning and learned
surrogate, and verify the Metropolis correction against the declared target.
Agreement with a fixed LEDH target alone does not establish agreement with the
original model posterior.

Read `docs/reference/hmc-tuning-interface.md` and
`HMC_TUNING_INTERFACE_CAPABILITIES` when wiring that comparison. The current
typed arbitrary position-only-force branch is classified as conditional
mechanics; it cannot silently issue an exact-score tuning authority. Resolve
that capability explicitly. Use adaptive warm-up and cumulative retained
sampling until declared R-hat, bulk/tail ESS and posterior MCSE criteria pass
or a predeclared sample/time cap is reached. Compare posterior means, scales
and quantiles with independent references. All valid retained samples remain
in the cumulative inference assessment; no extra fixed-size validation chain
is proposed here. Acceptance is explanatory, not proof of convergence.
