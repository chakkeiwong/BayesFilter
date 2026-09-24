# Bounded continuation: isolate the relative-shape solver limit

This is the smallest next diagnostic after the relative-shape campaign at
clean source e6298503. The owner has authorized continued master execution.
Launch 2 completed with 29 accepted rows, one unconverged validation row, no
underflow rejections, and two blocked weak-regime claim rows. The curved claim
has boundary activity and larger observed score error than EKF, UKF and the
local-linear proposal; it is not promoted. Those are candidate failures, not
evidence of an invalid harness or rejection of iAPF.

Question: does increasing the solver cap alone resolve the saved weak-regime
validation failure (dataset 1212, iteration zero)? Baseline is the stored
2,000-step fit, whose projected gradient was 1.972863e-5 against tolerance
1e-7, with finite coefficients and relative residual 3.595909e-8. Compare one
10,000-step arm, the existing implementation's maximum. Keep objective,
initialization, line search, bounds, tolerance, model, observations, offline
random streams, dtype, TF32, XLA, GPU class and floor unchanged. This tests a
solver budget hypothesis. It does not retune or rerun any claim dataset.

Primary pass criterion: the replayed first iteration and entire finite iAPF
procedure meet the unchanged convergence and validity checks. Verify identical
first offline seeds and first likelihood to the failed validation record before
interpreting the comparison. Failure to do so invalidates the diagnostic.
Finite coefficients, covariance, cast, reference refinement, device growth and
source/driver provenance are validity vetoes. Projected gradient, optimizer
steps and residual explain the result. Score error and runtime cannot nominate
or promote a setting from this replay. The four heuristic results from launch 2
remain promotion veto evidence for that candidate; this diagnostic cannot erase
them. No new score ranking, default, scientific, HMC or LEDH claim is made.

The 10,000 cap comes from the existing bounded fitter API, not a recommended
default. It may merely spend longer in a poorly conditioned valley. The cheap
discriminator is the unchanged stopping norm, which distinguishes insufficient
iteration allowance from accepted convergence. If it still fails, investigate
optimizer geometry before another calibration; do not lower the threshold. If
it succeeds, a future fresh calibration must evaluate a justified solver budget
and still address the Gaussian-family/boundary and heuristic failures.

Skeptical audit passed for this narrow question: exact same validation data and
first cloud are legitimate for diagnosis, not a fresh validation claim;
increased work is explicit; numerical tolerance is unchanged; failed claim
data 1220/1221 are not used; no proxy residual is promoted to score quality.
The Algorithm-3 objective and derivation are in
`younis-score-iapf-relative-shape-repair-2026-09-17.md`.

Budget before this action: 246/280 charges, two of three GPU campaign launches,
at most 226.98/3000 GPU wall seconds, and about 1005.80/7200 CPU seconds before
additional documentation allowances. One row costs at most four charges; no
additional GPU campaign launch is available afterward. External timeout is
600 seconds plus 20 seconds for termination. This remains inside all original
limits. This diagnostic finishes the campaign's launch allocation; a later
fresh calibration needs a new bounded campaign plan.

Execute the preserved diagnostic driver in
`artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-relative-shape-repair-20260917-01/solver_budget_diagnostic.py`
from clean source e6298503, using tftwogpu, UUID-selected RTX 5080, trusted GPU
access, verified memory growth, and no package/environment changes. The driver
hash, exact argv/environment, source fingerprint, input study, failed/successful
result, comparison and timing go into a new `launch03-solver-diagnostic`
directory. Candidate failures remain saved. The decision and remaining budget
will be reported alongside launch 2, and the master/checkpoint updated.

Executed: the first fit converged after 4,201 steps with projected gradient
9.884881624344644e-8; the complete replay passed. First offline streams and
likelihood matched exactly. This consumed three charges and the third GPU
launch, closing the campaign at 249/280 charges. The 2,000-step allowance was
insufficient for this case; no score/default promotion follows from the replay.
