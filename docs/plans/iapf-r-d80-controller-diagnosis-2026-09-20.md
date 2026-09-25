# Diagnose the frozen d80 controller failure

This is the failure-analysis branch of the authorized log-fit validation plan,
not a new compute allocation. Reserve at most200 R-worker seconds and one of
its eight launches, under the same1800-second total. Output:
`artifacts/iapf-r-log-fit-validation-20260920-01/attempt06-d80-diagnostic`.
The valid d40 follow-on proceeds independently. The planned fresh d80 follow-on
is blocked by the incomplete d80 pilot.

Observed: d80 seed72000080, method seed61000011, iteration cap20; particle
history1000/2000/4000/8000. All2000 fitted regressions pass, maximum QR normal
equation residual8.527e-16. The last six likelihood estimates have CV1.401898,
above the fixed0.5 stopping threshold. Their spread includes about36 log units.
These are adaptive training estimates, not independent final estimates.

The inherited20-iteration/N16000 safety limits came from the original bounded
replication plan (line136). The new validation assumptions should have listed
them explicitly. They bound compute; they do not come from a proof that the
paper's algorithm must stabilize within20 iterations. We retain both limits
for this diagnostic. Hitting one establishes incompleteness within the local
budget, not mathematical impossibility of convergence with more computation.

Question: does an exact replay expose weight concentration or prefix errors
while the QR regressions remain admissible, and does the exact-twist control
still recover Kalman on this very dataset? Baseline is the saved failing RDS
and the exact Kalman likelihood. No fitting/control setting changes.

Implement a diagnostic consumer of the same captured `iapf_iterate`,
`iapf_apf` and backward fitter. Wrappers retain already-computed ESS,
resampling/prefix errors and floor diagnostics at each adaptive pass, and save
the twists and RNG state before each call for targeted subsequent replay.
They perform no random draws and must reproduce every saved likelihood and
particle count exactly. Pass/fail is this exact replay plus the full-covariance
exact-twist control's absolute log-likelihood error<=1e-7 on the same data.
The oracle bypasses fitting and cannot promote the fitted candidate.
ESS and training/prefix errors are explanatory only. A mismatch, missing rows,
nonfinite value or failed oracle invalidates the diagnostic. Reproducing the
iteration cap is expected, not a reason to terminate before recording evidence.

Default audit: the data/method seeds and method controls are frozen from the
failed pilot; using the same data is required for diagnosis and excludes
holdout inference. N16 for the exact-twist oracle is a mechanics convenience:
the theoretical weights are constant for every N, so its role is an algebraic
control, not an optimized comparator. The1e-7 oracle tolerance is conservative
against the prior1.82e-12 errors and cannot be used for fitting accuracy.
No clipping, regularization, cap extension or altered objective is introduced.

Skeptical review: longer blind replication would confound controller resource
limits with bad fits. This replay separates executed-trajectory fidelity,
Gaussian filter algebra and particle-weight behavior. It cannot prove that
diagonal covariance or the log loss alone causes failure, nor reject the
original author's implementation. Record actual time in the common budget,
save the exact command/environment/sources and update the terminal result with
the failed candidate as a headline. Review passes for this bounded diagnosis.
