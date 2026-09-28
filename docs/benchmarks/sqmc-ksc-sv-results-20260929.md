# KSC mixture comparison and terminal review — 2026-09-29

Configuration: FP64 TensorFlow/GPU/XLA diagnostic comparison, TF32 off, N=1,008, Contract E and dual-cap safeguards enabled. Every route/horizon has separate calibration, frozen controls and disjoint validation. The production FP32/TF32 program was not tested. The target has one latent state and two parameters, theta=(gamma_raw,log_beta)=(1.5,0), Q=1, h0~N(0,1), and transition before observation.

**At T=120, the Gaussian Kalman approximation has lower observed score error than each of the four particle methods in six of eight pairs.** Its mean score-vector error is 0.49483, compared with 1.54362–1.60816 for the particle methods. This is a promotion veto for those cases. All 128 final particle evaluations nevertheless passed the declared validity checks, and all four methods remain research candidates under the owner’s retention decision. No overall method winner or default promotion is established.

The score and likelihood findings differ. At T=120, Gaussian Kalman has mean absolute log-likelihood error 9.41472, versus 1.28587–1.68852 for the particle methods. A method can approximate the scalar likelihood more closely while its derivative is less accurate.

## Comparison

Entries are mean score-vector L2 error ± standard error of that mean, across eight independent dataset/design pairs. The accuracy reference integrates the full seven-Gaussian-mixture likelihood. Gaussian Kalman replaces that mixture by its moments and is an approximation, not an exact oracle.

| T | IID | Inverse CDF | Permutation .98 | Permutation .97 | Gaussian Kalman |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.21330 ± 0.04418 | 0.16295 ± 0.04293 | 0.16360 ± 0.04276 | 0.16504 ± 0.04311 | 0.38244 ± 0.12801 |
| 20 | 0.35479 ± 0.08668 | 0.27694 ± 0.06177 | 0.27720 ± 0.06212 | 0.27645 ± 0.06307 | 0.41078 ± 0.09467 |
| 50 | 0.46267 ± 0.08997 | 0.56875 ± 0.12536 | 0.57420 ± 0.12520 | 0.57172 ± 0.12599 | 0.47941 ± 0.08081 |
| 120 | 1.54362 ± 0.40666 | 1.60816 ± 0.43979 | 1.58895 ± 0.43452 | 1.60149 ± 0.43352 | 0.49483 ± 0.16550 |

The [full report](../plans/artifacts/sqmc-ksc-sv-20260928/final-evidence-01/report.md) includes sample SD, SE, actual mean scores, coordinate absolute errors, actual log likelihoods, all 24 paired intervals and conditional heuristic comparisons. [scores.csv](../plans/artifacts/sqmc-ksc-sv-20260928/final-evidence-01/scores.csv) preserves every individual score coordinate and its reference/error; [values.csv](../plans/artifacts/sqmc-ksc-sv-20260928/final-evidence-01/values.csv) preserves every likelihood.

At T=10, the paired 95% t interval for IID minus inverse-CDF mean error is [0.009670,0.091038]. The corresponding IID-versus-permutation intervals also exclude zero. These are exploratory pairwise findings without multiplicity correction, conditional on this model regime and selected controls. Every interval comparing the three SQMC variants includes zero at every horizon; all route-comparison intervals include zero at T=20,50,120. There is no statistically supported overall ranking.

For e_i=||score_i−reference_i||_2, SD(e) measures variability between pairs, while SE(mean(e))=SD(e)/sqrt(8) estimates the standard deviation of the sample mean. For example, inverse CDF at T=120 has SD=1.24391 and SE=0.439788. These include both data and design variability. Conditional Monte Carlo uncertainty for one fixed dataset was not estimated. Signed coordinate means and their SEs are recorded separately; the norm of the mean error vector differs from the mean of the error norms.

## Material changes

- Replaced the wrong Gaussian-oracle assertion with an independent mixture-grid likelihood/score recursion, checked against exact T=1,2 mixture enumeration and finite differences. Historical oracle notes now carry correction notices.
- Corrected the missing +2*d(log_beta) term in the canonical KSC observation-flow tangent. The earlier tangent was wrong for the stated total derivative. Removed NumPy scalar constants from this touched KSC execution path.
- Added a KSC adapter that uses the shared canonical analytical executor, KSC-specific tuning-scope metadata, and a bounded campaign runner with shared accounting, source snapshots, complete logs and separate failure classifications.
- Added actual-coordinate reporting, SD/SE, paired intervals, conditional heuristic checks and dedicated calibration-data safeguard sensitivity. No final data selected controls.

## Terminal review

The claimed target is the likelihood and two-coordinate score of the repository’s seven-component KSC mixture. The reference computes a discretized integral and its analytic normalized-density derivative. It is an empirically converged approximation to that target. Gaussian Kalman computes a different, moment-matched likelihood; the KSC mixture itself also approximates native log-chi-square SV. Neither native-SV exactness nor long-horizon mathematical exactness is claimed. The particle score is the checked analytical derivative of its finite value program; its observed discrepancy from the mixture score is reported, not hidden by the finite-program derivative check.

Five CPU tests passed. All four complete GPU consumers passed tiny fixed-input finite differences and graph/XLA parity, then all-coordinate N1008 checks. Those preliminary checks were followed by 16 independently tuned route/horizon scopes, 128 valid final particle evaluations and 256 particle score coordinates. There were zero invalid candidate scopes, zero reference failures and zero infrastructure retries. Large finite errors are accuracy findings, not infrastructure failures.

Every dataset met the predeclared grid-refinement tolerances. The largest observed discrepancy was 5.11591e-13 in log likelihood and 4.60743e-14 in a score coordinate. Refinement agreement is not a rigorous error bound.

All 96 safeguard perturbations remained valid. The largest absolute coordinate change was 0.457833 for trust radius, 0.0482152 for LM damping and 0.00432089 for reset ridge. These diagnostic perturbations did not select settings or establish non-harm. Every scope selected epsilon=102.4, the upper end of the transport ladder; flow resolution was selected from only 2 and 8. The limited control family and sensitivity remain material explanations for poor long-horizon accuracy. Validation checked finite, consistent outputs; it did not certify accuracy.

The report assembler checked devices, memory growth, precision, tuning scope, seed separation, shared observations/references, actual coordinate errors and source snapshots. A separate arithmetic pass reconstructed errors from actual scores and checked all 24 summary SD/SE calculations and all 24 paired intervals with explicit sum-of-squares formulas. Input hashes and all 770 current source files matched the archived snapshot. Peak TensorFlow allocator usage was 1,078,209,536 bytes; this is allocator peak, not NVIDIA process reservation. Control-specific graph creation explains the preserved TensorFlow retracing warnings; each candidate kernel has a stable input signature and a bounded cache.

This was a local Codex plan and terminal review with executable checks. No independent reviewer was launched. The strongest alternative explanation for an apparent advantage is the single persistent regime, eight random pairs and restricted scope-specific tuning. A reversal on fresh regimes or replications would overturn a general ranking. T=120 heuristic losses question the tested configurations, not the validity of the harness or the entire research direction.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain all four methods | Completed declared likelihood/score comparisons | Validity passed; conditional heuristic promotion vetoes remain | Eight pairs, one regime, finite N/control family | Fresh tuning and replication before selection | No method elimination or general winner |
| Accept bounded campaign evidence | Actual scores/errors and uncertainty preserved | Reference and provenance checks passed | Empirical quadrature convergence; local review | Broader regimes and particle/resolution convergence | No native-SV, production or HMC admission |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No nonfinite or failed-validity final cells; Gaussian score losses in 60/128 comparisons, including 6/8 for every route at T120, veto promotion in those cases |
| Statistically supported ranking | Exploratory T10 pairwise intervals favor each SQMC route over IID; no SQMC ordering or overall ranking |
| Descriptive-only differences | Other observed means, likelihood errors, safeguard changes and timing |
| Default readiness | Not established; all numerical controls remain scoped hypotheses |
| Next evidence needed | Fresh T120 tuning for transport/flow resolution, N2016 convergence, more independent pairs, fixed-dataset design replications and broader parameter regimes |

## Accounting and completion

The KSC GPU-owning checks and workers consumed a conservative 2733.532536 seconds (0.759315 hours), including CPU-reference and compilation overhead inside those workers. Aggregate charge is 6.780795/12 GPU-hours, including prior work and the unchanged 300-second provisional prior-hook reserve. Remaining aggregate budget is 5.219205 hours. The elapsed renewal deadline remains 2026-09-30T16:15:40.010888+00:00.

The bounded plan is complete. N2016, sixteen-pair confirmation and broader regimes remain deferred studies, explicitly excluded from this stage. No HMC, package/environment change, push, publication or scientific/default promotion was performed. Complete commands, environments, seed partitions, source hashes, raw logs, tuning traces, observations and results are preserved under the versioned attempt-02 directory. The run starts from commit 479a4616 plus the archived source snapshot 36b0541d1d6409cc5d282cf960352500c42941cb341ad7e1470b14f630b65b40.
