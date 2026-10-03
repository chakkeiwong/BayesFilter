# Isolating covariance projection and the positive floor

Status: COMPLETE. 419 terminal checks pass; see
[results](artifacts/iapf-covariance-floor-factorial-20260922-01/result.md).
Authorized continuation of the renewed 48 CPU-hour and 48
GPU-hour campaign. Phase12 is complete; its result rejects promotion of the
diagonal, .01-floor score-regression candidate, not the fitting mechanism.
Root: `docs/plans/artifacts/iapf-covariance-floor-factorial-20260922-01/`.
Starting budget: 45.832301 CPU /47.767242 GPU hours. This phase allows at most
one hour per resource and four launches of at most 600 seconds including repairs.

## Question, mathematical controls, and evidence contract

Which of two retained choices explains the candidate's downstream failure:
discarding fitted correlations, or adding a constant to the Gaussian density?
Run the full factorial: covariance {diagonal, full} times floor {.01 times the
Gaussian peak, negligible with log ratio -1000}. Refit the complete backward
recursion for each arm. Replacing only the final covariance or floor would
confound this question, because each changes all preceding targets.

The score regression, whitening, rank and precision checks are unchanged from
phase12. With full covariance and a negligible floor, every backward target is
Gaussian. Its coordinate score is affine. For a full-rank whitened cloud the
regression therefore recovers the exact Gaussian coefficients at every time,
by backward induction. This predicts oracle equality independently of sampling
quality. N=256 exceeds d+1 for all three dimensions. No oracle coefficient is
fed into the fitting rule. At positive floor the target is generally not
Gaussian; affine score regression then remains an approximation.

The floor is rho times the normalized Gaussian peak:
`log_floor = log(rho) - d*log(2*pi)/2 - log(det(V))/2`.
Full covariance requires the determinant, not the product of marginal variances.
The positive arm preserves the realized phase12 scalar rho=.009999999776482582.
The -1000 arm is a Gaussian-limit diagnostic. It is not a proposed production
floor or a numerical robustness default.

Use exactly the phase12 12 data sets, T=8, N=256, dimensions 2,5,10 and seeds
92101:92104. Regenerate identical bootstrap pilot clouds and eight final streams
from its saved seeds. Verify phase12 manifest hashes before consuming inputs.
Preserve observations directly. This is an explanatory replay, not a fresh
holdout, tuning run, Algorithm4 adaptive experiment, or paper-model replication.

Primary engineering criteria (continuation veto on failure, then local repair):

- Original diagonal/.01 coefficients and finite likelihoods replay within 1e-8.
- Full/negligible coefficients equal the full oracle within 1e-8; all 96 final
  values equal the independently checked finite-initial-integration predictions
  within 1e-8, with postinitial ESS/N within 1e-8 of one.
- Independent base R reproduces all accepted recursive coefficients to 1e-8
  and agrees on rank/precision rejection. One trace per fixed TF configuration.
- Artifacts, finite quantities, source provenance and data identity remain valid.

Primary downstream quantity: mean squared log-likelihood error against exact
Kalman, eight paired streams averaged within each data set and then conditionally
within dimension. Report all four arms and the preserved heuristic ladder:
bootstrap (no learned guide), exact current-observation guide (myopic information),
full backward oracle (conditional ceiling with finite initial quadrature).
The constructed heuristic set and conditional situations are unchanged from
phase12. Observed loss to any heuristic is a promotion veto, never a continuation
veto. Comparisons use equal particles, not equal runtime; no speed claim follows.
Four reused data sets per dimension cannot establish a population ranking.
Paired differences and interactions are descriptive mechanism evidence only.
Report machine-readable heuristic verdicts and preserve per-data-set quantities.
The full/negligible versus full-oracle comparison is an identity check under
the declared 1e-8 numerical tolerance, not an empirical ranking. If that identity
passes, classify this pair as the same finite program; do not turn roundoff into
a heuristic loss. All other heuristic comparisons retain their declared rule.

Rank/precision/nonfinite rejection is a candidate veto; keep its rows instead
of silently omitting them. Shape residual, symmetry, whitening margins, ESS,
Gaussian-mixture probability and fitted-guide Gaussian KL are explanatory only.
No small residual or oracle equality alone promotes a fitted nonlinear method.
No paper replication, original-author fidelity, model-score correctness,
canonical LEDH, default, HMC or general nonlinear claim follows from this phase.

## Default audit and skeptical review

| Choice | Provenance and status | Justification | Failure mode and earliest diagnostic |
|---|---|---|---|
| Fixed local LG model and saved clouds | Phase12 explanatory baseline | Isolates the two factors | Not paper H=I; report actual H and prohibit transfer claim |
| Full covariance | Explicit extension | Regression already recovers it | Wrong normalization; independent R and Gaussian oracle equality |
| Diagonal covariance | Phase12 baseline | Exact replay comparator | Loses correlated geometry; factorial measures conditional effect |
| .01 floor | Inherited hypothesis | Reproduces failed candidate | Suppresses Gaussian mixture in dimension; record mixture probabilities |
| Log floor ratio -1000 | Mathematical control | Approximates zero floor without changing consumer signature | Invalid arithmetic; finite and oracle identities checked |
| No ridge or precision clipping | Same derived Gaussian fit | Exact full-rank recovery; rejecting invalid cases preserves target | Nonpositive precision on non-Gaussian targets; record margins and reject |
| T8,N256,four data sets,eight streams | Fixed convenience diagnostic scope | Reuse isolates mechanism with bounded cost | No population or paper-scale inference |
| FP64 GPU/XLA, TF32 off | Independently checked reference backend | Avoid known unresolved TF32 comparison veto | Device/growth/trace provenance; baseline replay first |

Skeptical review PASS before execution. The full oracle is a mathematical control,
not a practical competitor or an unbiased zero-variance claim: the consumer still
integrates X0 with finite particles. This phase controls the previously unfair
covariance/floor differences explicitly. No tuning or retrospective promotion
criterion is introduced. Reused data, correlations in the local observation
matrix, one-pass fitting and population uncertainty remain explicit limitations.
If the exact Gaussian arm fails, stop interpretation and diagnose implementation;
if an approximate arm fails, retain it and continue the mechanism analysis.

## Execution and review

1. Extend only the diagnostic recursive factory with explicit covariance/floor
   modes, preserving the default diagonal/.01 arithmetic. Implement a driver
   with source snapshots, input hashes, saved seeds, bounded attempts and budget.
2. Run GPU factorial on RTX4080SUPER UUID
   `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, with escalation, verified memory
   growth, float64, TF32 off, stable signatures and XLA. Save complete logs.
3. Run independent base R recursion and assemble conditional results on CPU with
   `CUDA_VISIBLE_DEVICES=-1`. Review numerical vetoes before interpreting errors.
4. Record decision/inference tables, source and run manifests, budget and concise
   next action. If covariance/floor explains failure, the next question is
   whether the paper's different H=I model has the same mechanism; do not infer
   paper failure from this stress model.

Commands (explicit working directory `/home/chakwong/BayesFilter`):

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_covariance_floor.py --mode gpu_factorial --attempt attempt01_gpu_factorial
/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_covariance_floor.py --mode cpu_reference --attempt attempt02_cpu_reference
```

Pre-mortem: apparent success could merely reflect exact-Gaussian structure or
the same reused data; preserve those boundaries. Failure could be determinant
normalization, a different pilot stream, or a non-SPD approximation rather than
evidence against score fitting. The baseline, R and oracle checks separate these.
Terminal review must distinguish interaction of factors from a single-cause
story and retain any rejected candidates. No further owner approval is needed
for unchanged-scope local repairs under the remaining campaign budget.
