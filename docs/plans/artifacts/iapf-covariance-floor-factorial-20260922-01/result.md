# The positive floor explains most of the large error on these cases

Keeping the diagonal guide and removing its .01-peak floor reduces observed
mean squared log-likelihood error from 1.064 to .022 at dimension 5 and from
15.230 to .081 at dimension 10. Restoring full covariance while retaining the
floor gives errors 1.237 and 12.456. The positive floor is therefore the larger
contributor on these preserved cases; covariance projection also matters. This
is a controlled explanatory replay of four data sets per dimension, not a
population ranking or new holdout evidence.

| d | Diagonal/.01 | Diagonal/negligible | Full/.01 | Full/negligible | Current observation |
|---|---:|---:|---:|---:|---:|
| 2 | 0.00990623 | 0.00790485 | 0.00855983 | 0.00657498 | 0.0143977 |
| 5 | 1.06441 | 0.0221285 | 1.23689 | 0.0125107 | 0.0310369 |
| 10 | 15.2295 | 0.0811927 | 12.4562 | 0.0392852 | 0.0692813 |

All 384 new final evaluations complete. The original diagonal/.01 coefficients
replay exactly. Full/negligible coefficients match the exact Gaussian oracle
within 4.89e-15, and finite likelihoods match within 2.85e-14. Independent R
reproduces all 48 recursive fits within 1.14e-13 and checks the oracle's finite
initial integration. All fixed TF/XLA configurations trace once. No oracle
coefficient enters the fitted candidate. The preserved bootstrap and full
oracle comparisons remain in the structured conditional table.

The mathematical mechanism is explicit. For a transition N(m,Q) and guide
psi(x)=N(x;c,V)+rho*N(c;c,V), its Gaussian mixture component has probability

    p = 1/[1 + rho*sqrt(det(Q+V)/det(V))
                 *exp((c-m)'(Q+V)^-1(c-m)/2)].

When Q=qV and m=c this becomes 1/[1+rho*(1+q)^(d/2)]. Thus a small fixed
peak-relative floor can dominate the integrated Gaussian exponentially with
dimension. The backward target gradient also multiplies its future-information
term by the corresponding Gaussian responsibility. The floor can weaken both
the proposal and the fitted lookahead. In this replay the mean Gaussian
proposal probability at d10 is .0622 for diagonal/.01 and .0436 for full/.01,
versus one for both Gaussian-limit controls. These observed quantities support
the derivation, but do not isolate the two pathways separately.

Full covariance and negligible floor restore an affine target score at every
backward step, so exact regression recovery follows by backward induction on
any full-rank cloud. The experiment verifies that finite-program identity.
Finite initial quadrature remains: this does not assert zero Monte Carlo
likelihood variance or general model-score correctness.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| Retain mechanism; do not promote a default | Replay, oracle and R identities pass | Positive-floor arms fail conditional heuristics; diagonal/negligible also loses to oracle | Population effects, paper model and adaptive behavior | Paper-model mechanism transfer with explicit timing | No author-code or paper replication claim |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Numerical validity passes for all four arms; observed heuristic losses veto promotion of approximate arms |
| Statistically supported ranking | None; four reused data sets per dimension |
| Descriptive-only differences | Conditional MSE and paired factor effects; no ordering generalized beyond these cases |
| Default readiness | No change; full covariance and score objective are diagnostic extensions |
| Next evidence needed | Paper model and timing, then adaptive/fresh likelihood validation under a declared floor hypothesis |

Post-run review: the local H contains cross-coordinate terms and differs from
the paper's H=I. Lowering a floor cannot repair the density objective's separate
degeneracy, and the successful full arm exploits Gaussian structure. The
strongest alternative explanation for transfer failure is this different model
and one-pass pilot design. A paper-model comparison is therefore required before
generalizing. Original author initialization and numerical floor remain unknown.
The next phase addresses this gap; no continuation veto fired.

The identity-equivalent full/negligible and oracle arms are classified as the
same finite program within 1e-8, not ranked by roundoff. This clarification was
recorded before terminal interpretation and is supported by the predeclared
identity check; other heuristic comparisons retain their original rule.

The plan, exact commands, source snapshots, seeds, CPU/GPU status and allocator
measurements are preserved in the versioned launch manifests. GPU work used
FP64 XLA, TF32 off, verified memory growth on the RTX4080SUPER; R intentionally
hid GPUs. All prior evidence remains unchanged.

Terminal verification: 419 checks pass. Remaining budget 45.832008 CPU /47.745145 GPU hours.
