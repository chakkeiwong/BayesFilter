# Phase 17 result: two distinct failure mechanisms and bounded remedies

Complete, 2026-09-23, in the isolated `ledh-refactor-with-policy-fix` checkout.
All eight selected phase 16 failures have successful diagnostic reruns: the four
capped learners require one more iteration; the four precision rejections each
require one explicitly recorded QR backward-fit rescue. On eight fresh healthy
cases across two datasets, enabling that rescue leaves the complete learner
result, guides and filter-call history bitwise identical. Original phase 16
failures remain failures; its 4992/5000 score completion count is unchanged.

This is independent R reference work on the paper's d80/T100 linear Gaussian
model. It is neither the paper's Equation 15 fitter nor a canonical LEDH result.
The [mathematical diagnosis](mathematical-diagnosis.md) derives the mechanism and
follows the actual consumer call chain. The reviewed frozen
[plan](../../iapf-score-failure-remedies-2026-09-23.md) defines this bounded scope.

## What failed and what the reruns establish

The positive constant added to a Gaussian guide changes the next backward
target's score to a nonlinear function. Its negative Hessian is
`G + r B - r(1-r) u u'`; the negative rank-one term can destroy log-concavity.
The score regression enforces symmetry but not positive definiteness. All four
first-fit failures reproduce exactly on full-rank clouds. Their fitted minimum
precision eigenvalues are -.2680, -1.1302, -.1402 and -.1652. This invalidates a
Gaussian fit; the fail-closed guard is working correctly.

The analytic curvature test finds 145, 289, 219 and 162 non-log-concave points in
the corresponding 1000-particle fitting clouds. Central differences of the
actual target agree with the score within 2.53e-8 and the Hessian-vector product
within 3.82e-7. On the SAME clouds and next mean/covariance, the floor-free
Gaussian score recovers the analytic Gaussian fit within 1.43e-14. This isolates
the floor/regression interaction. It does not prove that every non-log-concave
cloud produces a rejected regression. Lowering the floor is not monotone:
label 291 initially gets worse. The floor also has a paper-motivated importance
sampling tail role, so it was retained in the evaluated rescue.

The old cap is a separate issue. Its loop executes zero-based iterations 0–11.
Each failed stopping history contains a high likelihood estimate at iteration 6,
which stays in the six-value CV window through 11. At iteration 12 it leaves:

| Label | CV at old limit | CV at iteration 12 | N | Earlier histories/guides |
|---|---:|---:|---:|---|
| 844 | .5426 | .3228 | 2000 | exactly identical |
| 891 | .5771 | .2932 | 2000 | exactly identical |
| 947 | .5838 | .1909 | 2000 | exactly identical |
| 989 | .7237 | .0951 | 2000 | exactly identical |

All four satisfy the unchanged .5 threshold with cap 18. Each still makes the
independent final APF call. This is a bounded-run truncation, not evidence that
the stopping test or iAPF itself cannot work. The paper does not specify our
12-iteration cap. Cap18 is a tested diagnostic bound, not a termination guarantee.

For each rejected score fit, the optional `score_qr_rescue` callback performs
the existing strict QR backward fit on the same clouds, records the rejected
score diagnostics, and subsequently returns to score fitting. All four selected
cases complete at iteration 7, N=2000, with exactly one rescue. A QR fit to the
exact failing local target is also valid in each case. The hybrid openly uses a
different fitting objective on rejection; it does not claim to repair the
unconstrained score objective or implement Equation 15. QR rejection remains a
failure; no eigenvalue clipping or hidden ridge was added.

## Fresh conditional probes

There are four independent learner seeds per new dataset, with original score,
the conditional rescue, standalone QR, three cheap controls, and the full
Gaussian oracle. All complete. No rescue is needed in these eight cases, and
the actual paired consumer results are bitwise identical. This tests non-harm
on healthy trajectories; eight cases cannot certify a rare-failure rate.

Observed relative likelihood RMSE, `sqrt(mean((Zhat/Z-1)^2))`:

| Method | Dataset 981080 | Dataset 991080 |
|---|---:|---:|
| Original score | .5115 | .1969 |
| Score with conditional QR rescue | .5115 | .1969 |
| Standalone QR | .1183 | .1314 |
| Bootstrap, N=10000 | 1.0000 | 1.0000 |
| Fully adapted, N=5000 | 2.7668 | .8901 |
| Current-observation guide, N=1000 | .9819 | .9859 |

These are descriptive, not statistically supported rankings. There are only
four seeds per condition; rare importance-weight tails remain a major
uncertainty. Both fitted methods pass the predeclared descriptive heuristic
screen on both datasets: all 18 paired candidate/control mean squared-error
differences are negative. This does not establish superiority. The exact oracle
is a correctness control, rather than a competitor an approximate method must
outperform; its maximum absolute log-likelihood error is 3.64e-12.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Precision mechanism identified | four exact replays, analytic/FD agreement, same-cloud intervention pass | negative precision confirmed; guard retained | finite-cloud geometry controls which fits fail | preserve diagnostics; keep rescue explicit | original-author iAPF fails |
| Cap extension resolves selected truncations | four exact prefixes/guides; stop at12 | original cap failures retained | future seeds can need longer or hit Ncap | use bounded continuation in future diagnostic plans | cap 18 always suffices |
| Conditional QR rescue remains viable | four selected recoveries and eight exact healthy identities | strict QR guards retained; no fresh completion veto | rare rescues and downstream tails unquantified | retain optional reference implementation; confirm before any broader promotion | default readiness or correctness of score-only fitting |
| Accuracy comparison remains provisional | conditional cheap-control screen passes | no observed heuristic loss here | four seeds/data cannot rank methods | preserve larger phase 16 evidence; specify any confirmatory scope first | QR superiority, population reliability or unbiasedness test |
| Paper/LEDH work remains open | Equation 15 identity and GPU/canonical gates not addressed | existing TF32 and source-identity limits remain | original solver details and transfer | follow master consumer revalidation, then source-grounded paper work | full paper replication, HMC or LEDH admission |

| Inference status | Finding |
|---|---|
| Hard veto screen | Original four nonpositive precisions are valid rejections; original four caps are incomplete results. All new required engineering checks pass. |
| Statistically supported ranking | None. Bitwise healthy identity is deterministic parity, not an accuracy ranking. |
| Descriptive-only differences | Conditional RMSE, individual rescued estimates, cost and particle counts. |
| Default-readiness | Not established. The new hybrid exists only in the diagnostic runner; original numerical sources are unchanged. |
| Next evidence needed | Fresh, predeclared rare-failure and downstream accuracy confirmation for any broader hybrid claim; separate Eq15 and TF32 work for the master goals. |

Post-run skeptical review: PASS for the stated local diagnosis and non-harm
question. The strongest alternative explanation is finite-cloud projection
geometry rather than pointwise curvature alone; both were measured, and the
same-cloud Gaussian intervention isolates the floor dependence. Selected
failures cannot estimate success probabilities. The weakest evidence is the
eight-case fresh sample, which almost certainly misses some rare events. An
unchanged healthy path producing a different result, a replay mismatch, or a
failed derivative check would overturn the engineering conclusion; none
occurred. A future fresh rejection would limit the hybrid's reliability without
undoing these exact local findings. No independent external reviewer was used;
the source/math audit and executable checks are the review evidence here.

## Reproducibility and budget

Commands actually run from the isolated checkout:

```
python3 docs/benchmarks/diagnose_iapf_score_failures.py --execute
python3 docs/benchmarks/summarize_iapf_score_failures.py
```

R 4.1.2, `/usr/bin/Rscript`, FP64 independent reference. Every worker deliberately
sets CUDA_VISIBLE_DEVICES=-1 and BLAS/OpenMP/MKL/Accelerate threads to 1. At most
two workers ran. Git base 89065bc6354801cb368d5e163ba49fd9c4372d10; exact new and
existing source hashes, plan, commands, environment, seeds, wall time and output
hashes are in `supervisor-01.json` and 45 `*-launch.json` files. Saved RDS inputs
include the generated model data. Failed-cloud contexts, fit diagnostics,
controller histories and final estimates remain in their unique directories.

All 45 tasks finished with exit 0; 64 reported likelihood records are present, and
275 output hashes were verified along with every recorded source hash. The
[verification](verification.json) contains exact checks and machine-readable
conditional heuristic verdicts. No new attempt failed or needed repair.

Charge: 784.1954409171594 single-worker wall seconds (13.070 CPU minutes), zero
GPU seconds, within the 7200-second phase cap. Remaining owner balance:
64923.84917570048 CPU seconds (18.0344 hours),171882.52032958285 GPU seconds
(47.7451 hours), no active reservations. The original phase 16 budget and raw
artifacts were not modified. This phase is complete; the wider master remains
open with its next action stated in the concise checkpoint.
