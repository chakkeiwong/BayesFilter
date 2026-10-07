# Oracle-free SIR tuning and protected-model regression plan

## Research question and intent

Can controls of the existing marginal-mixture LEDH reduce repeated-design
variation and conditional reset distortion in the SIR d=18 log likelihood and
analytical score, without changing LGSSM, KSC SV, or predator-prey results?
This is a bounded diagnostic campaign, not a claim that a full SIR oracle is
available. Every filter evaluation uses the canonical shared Contract-E
analytical recursive score. No global or HMC defaults change.

The exact comparator is `run_ledh_nonlinear_master.arm_settings("guarded_pairwise",
BASE)` with `importance_weight_policy="marginal_mixture"`. It retains pairwise
steps 4, moment safety on, normal-quantile residual design, and coordinate-cap
identity radius 8. The first draft accidentally used pairwise steps 0 and safety
off; that wrong-baseline defect was repaired before serious execution.

The scope is SIR, T=10/20/40/50 separately, N=1008, FP64 TensorFlow/XLA,
TF32 disabled, GPU memory growth, route iid_dual_cap, exact-divisor chunking.
Changing any scope field requires new tuning. These results cannot transfer to
a different model, horizon, precision, particle count, or data regime.

## Evidence contract

- Primary nomination criterion: after validity checks, calibration point
  estimates of log-likelihood variance, all three score variances, and conditional
  reset RMSE for log likelihood and all three score displacements must be at most
  1.10 times the comparator. Minimize the sum of four standardized variances
  only among eligible candidates. The mean likelihood is never optimized.
- Validation and confirmation: evaluate the frozen nomination without reselection.
  Report the same ratios, paired mean differences, MCSE, and intervals. A point
  screen is not a statistical proof of non-inferiority.
- Promotion vetoes: numerical invalidity; analytical/finite-difference mismatch;
  failure of value/score variance or reset screens; observed underperformance
  against a heuristic; any protected-model regression; absent/stale scope metadata.
- Continuation vetoes: missing trace fields, broken reference or source wiring,
  artifact corruption, exhausted budget, or a protected-model regression.
  A poor candidate remains rejected but does not cancel predeclared diagnostics.
- Repair triggers: localized runner/resource failure permits a fresh numbered
  retry in the same budget. A score/FD mismatch requires localization and prevents
  an exact-score claim; it cannot be hidden by low variance.
- Explanation only: particle ESS, cap activity, runtime, and transport diagnostics.
  These do not establish accuracy. Local reset error is an explicit nomination
  criterion but cannot by itself establish full-filter accuracy.
- Not concluded: exact SIR likelihood/score accuracy, repaired accumulated bias,
  universal mixture dominance, HMC readiness, or production-default readiness.

Artifacts are under `docs/plans/artifacts/ledh-sir-no-oracle-tuning-20261006-01/`:
source checksums plus Git commit, exact commands, environment, GPU status,
observations/hashes, seeds, wall time, append-only rows, selections, full values
and scores, per-step reset diagnostics, and result/decision records.

## Mathematical diagnostic

For the actual weighted pre-reset cloud mu=sum b_i delta(x_i) and uniform
post-reset cloud nu=N^-1 sum delta(z_i), compute
h_theta(x)=N(y; H F_theta(x), [1+100 exp(2 theta_sigma)] I_9).
This is an exact one-step convolution for the repository SIR model (Q=I_18,
HH'=I_9, R=100 exp(2 theta_sigma) I_9). It is conditional on the finite clouds.

Compute Z_mu=sum b_i h(x_i), Z_nu=mean h(z_i), and log Z_nu-log Z_mu.
All three total derivatives include cloud tangents and pre-reset weight tangents.
The score displacement is
(D Z_nu - D Z_mu - s_mu (Z_nu-Z_mu))/Z_nu.
The runner evaluates it as the difference of normalized total derivatives.
Log densities and log-sum-exp avoid artificial density floors and underflow.
Log Z_mu, log Z_nu, log errors, score errors, and relative ESS are preserved.
Absolute and relative density errors can be reconstructed where representable.
This check isolates one realized reset; it is not a full SIR filtering oracle.

## Candidates and independent partitions

Six hypotheses use the same random inputs within each partition:

1. Exact guarded marginal baseline (flow 8, epsilon 102.4, strengths .12/.03).
2. flow4: four flow substeps.
3. epsilon51: epsilon 51.2.
4. weak_correction: strengths .06/.015.
5. epsilon204: epsilon 204.8.
6. flow16_weak: sixteen flow steps and strengths .06/.015.

Every other control remains fixed. Factor-of-two changes are sensitivity
hypotheses, not justified universal settings. The existing cap remains on;
its potential predictive distortion is measured, not assumed harmless.

Observation seeds: calibration 26100611, validation 26100612, confirmation
26100613. Related prefixes share the same seed within a partition; horizons
are reported separately. Particle-design seeds: 261006211--214 calibration,
261006311--314 validation, 261006411--418 confirmation. Both types of seed are
disjoint across partitions. Selection is written before validation is opened.
Validation cannot reselect. Confirmation evaluates the frozen nomination even
if validation rejects it, to measure generalization without tuning on holdout.

Paired mean-difference t intervals and paired bootstrap variance-ratio intervals
(2,000 resamples; reporting seed 261006901) are reported. Four/eight replicates
and one dataset per partition limit power and generalization. No superiority
claim follows from a calibration minimum or a point variance ratio.

Constructed heuristic adversaries: ancestor_guarded (original component ratio),
covariance_only (no higher-moment fitting), diagonal_only (no pairwise fitting).
Each uses the first two confirmation seeds and is compared with the selected
candidate on those same seeds. They cannot influence selection. Situations are
short/long horizons plus per-time-step reset and ESS traces, exposing late
collapse. Observed deterioration vetoes promotion; two seeds cannot establish
superiority. Exact LGSSM/KSC reference checks are separate.

## Protected-model preservation

Run focused canonical marginal-weight, analytical-score, LGSSM reference, and
KSC mixture-reference tests. Replay LGSSM, KSC SV, and predator-prey at each
T=10/20/40/50 through the clean-HEAD and current public evaluator with identical
inputs and controls; preserve values, every score coordinate, and differences.
This is an implementation-preservation check, not proof of nonlinear accuracy.
The only shared-code modification exposes already-computed arrays through an
opt-in diagnostic trace. SIR-selected controls are not used by other models.

## Assumption audit and pre-mortem

| choice | provenance and reason | failure mode | early check | status |
|---|---|---|---|---|
| guarded marginal baseline | exact completed-campaign configuration | wrong arm silently weakens comparator | import arm_settings; check manifest | frozen baseline |
| N=1008, FP64/XLA | current comparison scope; divisible by 2d | no transfer to other scopes | manifest and directional equality | scope only |
| normal quantiles | current guarded residual design | design-dependent behavior | record identity; no cross-design claim | frozen baseline |
| 10% margin | engineering diagnostic guardrail | noisy point ratios appear conclusive | intervals; no automatic promotion | hypothesis |
| moment safety on | existing tail guard | predictive distortion despite finite outputs | exact local reset changes/cap trace | frozen baseline |
| factor-of-two grid | bounded sensitivity search | misses useful controls | report rejected/viable arms and grid limits | hypothesis |
| 4/4/8 seeds | bounded compute, heldout confirmation | low statistical power | MCSE and paired intervals | descriptive limits |

A low-variance biased cloud could pass. The local predictive check detects some
reset distortion but cannot certify accumulated bias. Mean shifts and all
coordinate scores must therefore be shown. A lower variance does not mean a
more accurate estimator. Scope-specific tuning must not leak to protected
models. The unchanged shared evaluator and exact replay test address that risk.

## Review, execution, and budget

Skeptical audit repaired three material defects before execution: wrong baseline,
observation partition reuse, and a missing D Z_mu term in the displayed score
identity. The revised plan includes exact comparators, explicit proxy roles,
heuristic checks, independent partitions, candidate versus continuation vetoes,
and no automatic default promotion. It passes as a bounded diagnostic campaign.
The review is self-review; it is not represented as independent peer review.

1. Compile/CPU smoke: N=72,T=3; all-direction trace/public parity and two FD widths.
2. Run focused CPU tests with GPU devices intentionally hidden.
3. Trusted GPU preflight verifies memory growth and XLA. Run protected replays.
4. Run calibration, freeze selection, run validation and confirmation; inspect
   all coordinate scores, failed screens, finite differences, and heuristics.
5. Record actual results and decision/inference-status tables, compile and inspect
   the monograph, update the checkpoint, and commit required files.

Worker budget: 28,800 seconds total; at most 7,200 per horizon/phase worker.
Failed attempts consume budget. Smoke/tests and protected replays have separate
30-minute bounds. No package/environment mutation is planned. Retry outputs use
fresh numbered directories; complete attempts are reused within this campaign.

Exact allow-listed command prefix:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter-SQMC/docs/benchmarks/run_ledh_sir_no_oracle_tuning.py`
Actions: `smoke --device cpu`, `tests --device cpu`, `preflight`, `regressions`,
`run`, `report`. GPU commands use trusted permissions. The user authorized
execution; the allow-list update groups local tool permissions without waiving
platform controls or scientific checks.
