# M13 independent likelihood-reference result — 2026-09-23

The full-trajectory importance reference passed the fixed-point likelihood
checks at all nine parameter settings. This closes the integrated reference
test for the declared T20 fixture and grid. It does not yet supply a parameter
posterior reference or complete the M13 surrogate-HMC test.

All 41 focused tests passed. Every T1 interval covered the independent
quadrature, every T20 particle ladder agreed, and every highest-N interval met
the .02 log-likelihood precision limit. The largest observed halfwidth was
.005539. Both proposal starts converged, with the required objective agreement.
Every finite positive interval from the earlier independent sequential
references overlaps the corresponding final interval; the first bootstrap
point's nonpositive lower interval is uninformative.

| Point | lambda_Q | lambda_R | log mean likelihood | Log interval halfwidth | All declared checks |
|---|---|---|---|---|---|
| 0 | -0.693147 | -0.693147 | -20.48895689 | 0.003880 | pass |
| 1 | -0.693147 | 0.000000 | 27.95144087 | 0.003950 | pass |
| 2 | -0.693147 | 0.693147 | 26.92550575 | 0.005539 | pass |
| 3 | 0.000000 | -0.693147 | 1.54157049 | 0.003963 | pass |
| 4 | 0.000000 | 0.000000 | 30.52713184 | 0.004808 | pass |
| 5 | 0.000000 | 0.693147 | 26.34322724 | 0.004004 | pass |
| 6 | 0.693147 | -0.693147 | 19.22682663 | 0.004295 | pass |
| 7 | 0.693147 | 0.000000 | 30.58225264 | 0.004519 | pass |
| 8 | 0.693147 | 0.693147 | 24.14571173 | 0.004067 | pass |

The latent position prior is an exact marginal of the state-space Gaussian
dynamics, with velocities integrated out. The fitted Gaussian and defensive
prior mixture only propose integration samples. Every sample receives its
full prior-times-observation-density divided by mixture-density weight.
The result is corrected integration, not a Laplace likelihood. The bounded
wrapped-density alias error is recorded separately.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Fixed-point likelihood reference qualified | All nine precision and particle-refinement checks pass | T1 oracle, finite estimates, proposal convergence and heuristic checks pass | Approximate Student-t coverage; unsampled parameter settings and tails | Existing-master M13 canonical analytical-force consumer, then scope-specific controls and posterior reference | Parameter-posterior correctness, surrogate-HMC success, default readiness or method ranking |

| Inference status | Finding |
|---|---|
| Hard veto screen | No declared veto fired in attempt03 |
| Statistically supported ranking | None |
| Descriptive-only differences | Earlier and final variance, ESS, interval widths and runtime |
| Default-readiness | Not evaluated; this is an independent reference |
| Next evidence needed | Canonical consumer, declared parameter prior/full support, controls, parameter reference and posterior chains |

The smallest final within-replicate importance ESS was 34,106 of 65,536 and
the largest normalized weight was .000250. These are useful diagnostics,
not proof that all tails or modes were found. Post-run red team: agreement
between two mode starts and three particle counts can miss a remote mode;
an independent full-horizon disagreement would reopen qualification. The
Gaussian-prior defensive mixture bounds weights, but finite variance alone
does not prove finite-sample interval coverage.

Attempt03: artifacts/surrogate-hmc-closure-20260922-m13-reference-03/.
Worker report: m13-reference-trajectory/report.json; command, git/source
hashes, seeds, environment, GPU memory-growth policy and charges are in
manifest.json, report.json and accounting.json. Sources remained unchanged.
Charged CPU41.051484/GPU29.533421 seconds; worker wall26.205587 seconds.
The three reference attempts used CPU106.771949/GPU162.927973 seconds.
Cumulative campaign charge CPU35445.538550/GPU33301.430671 seconds.
Remaining campaign CPU18554.461450/GPU193498.569329 seconds.
No fourth reference attempt is authorized within this completed pass.

Command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-reference-trajectory --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-03

Next is part of the existing master: trace and integrate the M13 canonical
analytical-force consumer. LG1 remains closed. The all-model master is incomplete.

## Preserved earlier attempt notes


The bootstrap reference failed the declared full-horizon precision criterion.
It did not test or reject surrogate HMC. This existing-master prerequisite
must supply an independent likelihood reference before posterior comparisons.

Attempt 01 used the actual state-4, observation-2, T20 range/bearing fixture
at all nine fixed pairs in {-log(2),0,log(2)} squared. Independent FP64
TensorFlow/XLA filters used 16 replications at N=4096,16384,65536. The
parameter coordinates multiply Q/R by exp(2 lambda); no parameter prior or
posterior-reference claim was defined.

All 35 focused tests passed, all T1 quadratures refined, and every highest-N
T1 interval covered its independent integral. Callback parity passed.
Every filter estimate was finite. The T20 log-likelihood interval halfwidth
exceeded .02 at all nine points: at nominal Q/R it was .0585766, and the
lowest-R cases were particularly noisy. A nonpositive lower interval at some
settings is an unresolved reference, not a demonstrated bias or disagreement.
The intervals average likelihoods before logging; they do not average log
likelihoods. Student-t coverage with 16 skewed replications remains approximate.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Bootstrap reference not qualified | All nine T20 precision checks fail | No nonfinite computation; T1 checks pass | Particle impoverishment and skewed likelihood estimates | Guided auxiliary importance reference within unchanged master scope | No surrogate-HMC rejection, posterior correctness or speed ranking |

| Inference status | Finding |
|---|---|
| Hard veto screen | Reference precision screen failed; model-law and T1 checks passed |
| Statistically supported ranking | None |
| Descriptive-only differences | Per-point variance, ESS and interval widths |
| Default-readiness | Not evaluated; independent reference only |
| Next evidence needed | Corrected guided proposal, fresh replicates, same nine-point/T1/T20 screens |

The reference repair uses a local Gaussian auxiliary proposal with its full
importance correction. The plan derives the cancellation of the auxiliary
weight and proposal density. It preserves the model and observations, uses a
fresh seed, and keeps the original thresholds. Guided implementation review
checked normalization, parent alignment, velocity cancellation, timing and
carried weights; focused numerical checks run before GPU execution.

Post-run red team: T1 correctness cannot establish T20 precision, and small
replicate variability cannot rule out missed likelihood tails. An independently
precise full-horizon reference could overturn the current precision verdict.
The weakest evidence is the skewed, 16-replicate uncertainty approximation.

Attempt 01 artifacts: artifacts/surrogate-hmc-closure-20260922-m13-reference-01/
(report.json, accounting.json, manifest.json, source snapshot and worker report).
Charged CPU31.712198/GPU65.571358 seconds; worker wall61.731760 seconds.
GPU memory growth and GPU/XLA placement recorded; allocator peak1,124,071,168
bytes. Captured sources unchanged. Command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-reference --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-reference-01

Next in the existing master: run m13-reference-guided in fresh attempt02.
The master is incomplete and execution continues.

Attempt02 (guided auxiliary reference) completed: 37 tests passed, sources
unchanged, all T1 checks passed, all estimates finite. Only point8 passed T20
precision at the highest N. Nominal halfwidth .0404715; lowest-Q/lowest-R
halfwidth2.0389822. It remains unqualified; comparisons of observed widths
are descriptive, with no supported method ranking. Charged CPU34.008267/
GPU67.823195 seconds. Evidence: artifacts/surrogate-hmc-closure-20260922-m13-reference-02.

Next in the existing master is the final allocated reference attempt using
a fully corrected joint-trajectory importance proposal. The plan specifies
its exact joint Gaussian prior, defensive mixture, unchanged checks and budget.
