# Localize the d80 floor failure before a repair

The owner requested continuation on 2026-09-20. This extends the existing
log-fit validation campaign, retaining its total 1800 R-worker seconds and
eight launches. Six launches used 1010.468207 seconds. Reserve at most 120
seconds for attempt07-floor-diagnostic; the final launch is available for a
subsequently reviewed repair within the remaining total. CPU-only R reference,
with CUDA hidden and one BLAS/OpenMP thread. No TensorFlow default change.

## Question and evidence contract

Does the chosen positive constant in the diagonal Gaussian guide make the
proposal effectively bootstrap at observation 93, causing the intermittent
d80 likelihood collapse? Saved calls 15,19,20 (iterations14,18,19) of attempt06
provide two failed passes and one near-Kalman pass. Reuse their exact twists,
particle counts and RNG states. No new fitting or controller limit changes.

Primary diagnostic criteria: instrumented baseline log likelihood must equal
the saved RDS history exactly and its prefixes/ESS must match saved CSV within
1e-8 absolute / 1e-10 relative tolerance. At every time save proposal mixture
probabilities, realized floor draws, incremental weight ESS, Gaussian overlap,
and each floor/guide's contribution. A replay mismatch, missing/nonfinite
required diagnostics or wrong captured source invalidates this attempt.

Compare the original floor against zero-floor counterfactuals at time93 alone
and at all times. These are diagnostic interventions, not candidates for
promotion: they remove the paper's lower-bound protection. They use the same
initial RNG, but changing mixture draws changes subsequent RNG consumption;
differences are descriptive, not paired statistical evidence. Exact Kalman is
the certifying likelihood reference. The previously checked exact-twist oracle
is an algebraic control. The constant-twist/bootstrap pass already failed by
3587.90 log units; use it only as a recorded explanatory baseline. No method
ranking or published replication is inferred from these chosen trajectories.

Constructed comparator set: original transition proposal (the floor component),
Gaussian-only proposal (isolates mixture damage), unchanged learned proposal
(controls for fitting), and exact Kalman. Evaluate ordinary observations and
time93 separately; do not pool the large failure into an average. This
mechanistic diagnostic cannot pass a heuristic-dominance promotion screen.

## Assumptions and source

The paper's Eq.16, local text lines925-944, requires positive c(N,m,Sigma) to
retain a transition-mixture component, without specifying c numerically. The
current choice is the Gaussian density at a chi-squared upper-tail probability
N^-2. Its provenance is our reconstruction, not original-author code. In a
Gaussian transition the floor probability is exactly c/(c+J), with
J=N(m;transition_mean,Q+V). A small c relative to the guide's peak does not
ensure small c/J. That is the failure hypothesis, not yet a conclusion.

Default audit: selected saved inputs are failure localization, not holdout
evidence; QR log fits remain a different objective from Eq.15. Zero floor is
only a limiting counterfactual with possible unbounded importance variance.
Also record analytic floor probabilities for powers2,3,4,8,16 on the unchanged
baseline proposal inputs. These powers span a calibration curve; they are not
selected by likelihood accuracy. A repair must retain positive protection and
justify its value separately. No off setting may silently become a default.

## Skeptical review and execution

Review passes: localized prefix losses, exact saved RNG and an analytic mixture
identity distinguish floor damage from a resource cap. No baseline replacement,
fresh-data claim or proxy-to-promotion inference is allowed. Zero-floor success
alone cannot establish a safe positive-floor repair; zero-floor failure leaves
guide misspecification and propagation instability open. Both outcomes trigger
the next discriminating repair, not rejection of iAPF or the author's results.

Command: `python docs/benchmarks/run_iapf_r_replication.py --campaign validation
--output docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01/attempt07-floor-diagnostic
--dimension 80 --data-seed 72000080 --mode floor_diagnostic
--fit-mode log_quadratic --timeout 120`.
The driver captures and executes sources and input RDS/CSV, preserves full
logs, commands, environment, seeds, SHA-256 and actual wall time. Result:
`attempt07-floor-diagnostic/results/summary.csv`, `proposal.csv`, `floor-curve.csv`,
plus a mathematical result note and an updated concise campaign checkpoint.
