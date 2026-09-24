# Gaussian-innovation controls for the short-horizon iAPF score

2026-09-18. Authorized continuation of the score master and the existing
resampling-control campaign. This is a local variance-reduction extension,
not a source-faithful iAPF/KDM algorithm claim or a canonical LEDH route.

## Question and mathematical target

Can Gaussian-innovation controls reduce the conditional mean squared error
of the terminal-genealogy Fisher statistic after ancestor controls? The target
is the physical observed-data score, grad_theta log p_theta(y_1:2). The finite
particle statistic S_N is only an estimator of that target: it can have
finite-N bias and is not the derivative of the executed particle likelihood.
The completed ancestor correction improved S_N but still lost to UKF on
dataset 1500 and was indistinguishable from no resampling on 1511.

Keep the five fixed datasets 1490, 1500, 1501, 1510, 1511, their model,
theta, N4096, T2 and frozen iAPF fits from the completed comparison. This
isolates the new control mechanism; it is not validation on new datasets.
Use fresh particle, calibration, final and reference-noise streams.

For each time (initial draw included) and state coordinate, let z_i be the
Gaussian innovations actually passed to the existing particle kernel. Let
z'_{mi}, m=1,...,M, be independent draws from exactly the same generator and
dtype. For h_1(z)=z and h_2(z)=z^2, define

    G_h = sqrt(N) [N^-1 sum_i h(z_i)
                  - M^-1 sum_m N^-1 sum_i h(z'_{mi})].

Conditional on the fixed observations and fits, the two empirical means have
the same expectation. Therefore E[G_h]=0 even if the generator's discrete
noise law has moments slightly different from an ideal normal. This identity
needs identical marginal laws and independent reference streams; it does not
require knowing those moments. Compute moments and reductions in FP64.
Ordinary arithmetic rounding remains a numerical limitation of the identity.
This subtraction does not change the innovations consumed by the filter.

With A the existing 12 ancestor controls and G the six new scalar controls,
independent calibration freezes B_A and B_AG before evaluation:

    S_A  = S_N - A B_A,
    S_AG = S_N - [A,G] B_AG.

Conditioning on calibration makes the coefficients constants, hence both
estimators have E[S_N], including its finite-N bias. Corrections stay outside
the self-normalized particle ratio. No oracle score enters either fit.
For independent reference samples, Var(G_h)=(1+1/M) Var(h(z)); choose M=16
to limit reference-centering variance overhead to 6.25% of that control's
variance. This is a cost/precision derivation, not MSE tuning. Conditional
integration and mixture-choice controls remain separate future mechanisms.

## Evidence contract and research intent

* Primary comparison: S_AG versus S_A, with the same 96 calibration runs and
  64 paired untouched final runs per dataset. Primary promotion within this
  diagnostic stage requires a negative upper endpoint of the paired squared
  error difference interval on each of the four nonlinear datasets.
* Use 40,000 paired bootstrap resamples and 99.75% two-sided percentile
  intervals (Bonferroni 99% family for four comparisons). Conditional on the
  realized calibration coefficients, these quantify final-stream uncertainty;
  they do not integrate coefficient-training or dataset uncertainty.
* Also compare the frozen protected ancestor coefficients from the preceding
  192-calibration confirmation, and raw Fisher. This prevents replacing a
  stronger existing baseline with a smaller fresh calibration unnoticed.
  Those comparisons use separately declared exploratory 99% intervals.
* Constructed cheap adversaries, separately in each nonlinear dataset:
  EKF score (first-order Gaussian approximation), UKF score (nonlinear moment
  approximation), and no-resampling particle score (avoids categorical
  ancestry noise). Exact Kalman is the affine sanity comparator. Recompute
  their values/scores through the existing consumer endpoints. Any observed
  heuristic MSE loss is a conservative promotion veto, with its uncertainty
  reported rather than described as a supported ranking automatically.
* Mean-bias screen: approximate Student-t 99% Bonferroni family over the 30
  score components in five datasets, plus the prior FP32 reference allowance
  5e-6(1+abs(reference)). Failure vetoes promotion, not research continuation.
  Passing does not prove zero bias. Control means/MCSE, regression singular
  values/rank, score variance, coefficient size, runtime and ESS are
  explanatory only. Do not tune controls or cutoff on any final quantity.
* Continuation vetoes: invalid/nonfinite outputs, reference mesh/domain/tail
  failure, source drift affecting the comparison, failed centering/wiring
  checks, seed overlap, unverified GPU memory policy, or exhausted budget.
  Candidate MSE losses are repair triggers for a later bounded investigation.
* No inference of universal improvement, default readiness, accurate long
  horizons, new-model validity, finite-N unbiasedness, an HMC force, completed
  fitting calibration, or LEDH admission. No runtime defaults change.

## Defaults, assumptions and pre-mortem

| Choice | Provenance and justification | Failure and earliest diagnostic | Status |
|---|---|---|---|
| Scalar sine/quadratic model, T2, N4096, five fixed datasets | Same scope as the finished campaign; validated quadrature is tractable | Results may not transfer; report each regime and forbid extrapolation | Diagnostic baseline |
| Frozen fits, theta and ancestor rule | Exact prior artifacts, hashes recorded; isolates score controls | Existing fitting bounds may harm performance; retain fitting debt and compare simple filters | Frozen baseline, not promoted fit default |
| h=z,z^2 at each of three times | First two moment fluctuations of a Gaussian proposal; six added controls | Weak correlation or coefficient overfit; fresh paired final errors | Local hypothesis |
| Independent reference M=16 | Center under actual noise law with 1/16 added variance | Reference correlations invalidate centering; disjoint seed inventory and finite-support enumeration | Derived precision/cost choice |
| 96 calibration, 64 final | Uses remaining budget across all regimes | Noisy fit and low power; same-calibration baseline, previous frozen baseline, intervals; inconclusive stays inconclusive | Bounded diagnostic, not default protocol |
| FP64 fit with explicit FP32 input precision | Previously validated rank safety derivation | Spurious near-null directions; singular values/rank and known-good regression | Optional checked safeguard |
| No regularization beyond rank cutoff | Ordinary least squares estimates covariance projection; changing its bias is not needed for the centering identity | Variance from finite calibration; final uncertainty and stored stronger baseline | Fixed baseline hypothesis, no final-driven tuning |
| GPU FP32/TF32/XLA filter; FP64 control math | Existing execution scope; unchanged particle kernel | CPU-only probe cannot certify GPU route; focused CPU checks then real GPU manifest | Existing diagnostic exceptions |

The run could appear successful solely because a 96-run baseline is weaker
than the previous 192-run baseline: retain the latter. An ideal-normal moment
assumption could shift the mean: use independent empirical reference centering.
A conditional reference fixes observations, not their population: state that
scope in the result. A negative result may reflect weak controls or finite
calibration rather than failure of iAPF or the Fisher identity.

Skeptical audit PASS, before implementation/execution: target and all baselines
are explicit; no score-variance or calibration metric substitutes for final
score MSE; adverse regimes are retained; candidate failure is distinct from a
continuation veto. Exact same-law centering resolves the material floating
moment concern. The call budget below fits the unspent campaign allowance
without opening a new campaign, changing hardware or expanding compute.
Self-review is recorded; no independent reviewer is claimed.

## Execution and budget

Repository /home/chakwong/BayesFilter, branch surrogate-hmc, starting HEAD
6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. Preserve unrelated changes.
Environment /home/chakwong/anaconda3/envs/tftwogpu/bin/python. CPU checks set
CUDA_VISIBLE_DEVICES=-1 before TensorFlow import; GPU execution is escalated,
with TF_FORCE_GPU_ALLOW_GROWTH=true and repository verification before device
initialization. All repeated numerical kernels have fixed signatures and XLA.

Remaining campaign allowance: one serious launch, 1116 filter calls,
1670.688405 driver seconds, three adaptive fits and 420 test/probe seconds.
This stage uses no new adaptive fits. Planned calls are exactly
5*(96+64) iAPF + 4*64 no-resampling + 5*2*6 moment-filter directions = 1116.
CPU reference integration and small unit checks are separately recorded.
Limit this launch to 600 driver seconds and tests/probes to 180 additional
seconds. A failed launch is preserved; no extra launch is silently authorized.

Commands (full actual invocations and timing also saved in logs/manifests):

    CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_younis_iapf_innovation_controls.py tests/highdim/test_younis_score_master_combinations_tf.py
    TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_younis_iapf_innovation_control.py --output docs/plans/artifacts/younis-iapf-resampling-control-20260918-01/innovation-confirmation01

Preserve the run manifest, hashes, actual seed pairs, frozen calibration rows
and coefficients before finals, per-replicate final errors, uncertainty,
machine-readable heuristic verdict, source-drift check and cumulative budget.
Terminal note: artifacts/younis-iapf-resampling-control-20260918-01/innovation-result.md.
Update master and concise checkpoint at completion, including candidate losses
and the exact next action. Broader fitting and LEDH work stays pending.

Pre-launch verification: 22 focused CPU/XLA checks pass in 12.05 seconds,
including the existing Fisher/genealogy regressions in addition to the two
test files listed above. GPUs were intentionally hidden. Finite-support
enumeration checks centering under a law whose mean and second moment are
not 0 and 1; a consumer wiring test checks the single existing filter call
and its actual noise inputs. Read-only preflight verifies unchanged prior
source hashes and 4,800 new seed pairs disjoint from all 17,920 pairs in the
preceding comparison and conditioning confirmation. Escalated nvidia-smi
confirms the RTX5080 is available with shared utilization; no timing ranking
will be made. No material scientific issue was found in the final self-review.

Terminal review, 2026-09-19: execution and arithmetic checks completed; see
[result](artifacts/younis-iapf-resampling-control-20260918-01/innovation-result.md).
All four nonlinear primary comparisons and observed heuristic screens pass.
Exact affine Kalman remains superior. The launch omitted the prior GPU UUID
pin, so it exposed both GPUs, with RTX4080 SUPER at TensorFlow GPU:0. This
unplanned execution-scope deviation prevents claiming intended RTX5080
replication. The exact executed source and this plan's pre-run version are
preserved. UUID pinning/identity checks and device observability are repaired;
6 focused CPU checks pass after the initial 22, but no repaired GPU launch
has run. All 4 campaign launches and 8000 calls are consumed. A new explicit
bounded plan is required for the pinned confirmation; this campaign is closed.
