# q20 training-gap investigation

The owner requests a detailed code/results investigation of weak NeuTra
whitening, including how three GPUs could improve the training process. The
question is which failures are demonstrated in the current q20/T30 training
path, which alternatives remain unresolved, and what smallest repair experiment
would discriminate them. Plain NeuTra HMC and the tempered NeuTra ensemble,
with identity latent mass, remain the scientific scope.

## Evidence contract and intent

Trace the actual training consumer through configuration, initialization,
batch-native target/score, RKL loss, optimizer, assessment, checkpoint resume,
map selection and GPU scheduling. Inspect the complete saved cohort as well as
the selected map. Compare each candidate with its own initialization and
distinct earlier checkpoints; cross-candidate losses on different banks are
descriptive, not a statistical ranking. Inspect stored map parameters and
diagnostic points for scaling/activation constraints. Match execution-source
hashes and separately record current-tree differences.

Expected failure modes are premature stopping, misleading progress telemetry,
optimizer noise/clipping, insufficient representation or scale range, incorrect
scores, weak selection, and unused GPU concurrency. The pass criterion for this
investigation is reproducible evidence for each reported gap and explicit
limits for hypotheses. Numerical invalidity, mismatched source/target/checkpoint
identity or conflicting input evidence veto the associated conclusion; they do
not reject NeuTra as a method. No candidate or default is promoted. Loss,
gradient norms, activation statistics and prior score residuals are explanatory
diagnostics and potential repair triggers. No posterior claim follows.

## Execution, bounds and assumptions

1. Read the saved source and complete cohort, plus current dispatch code. Use
   standard-library/explicit diagnostic analysis without importing the target.
   Preserve input hashes and machine-readable per-candidate summaries.
2. Obtain one trusted read-only inventory of all GPUs. Existing GPU timing is
   historical scope-specific evidence; device count alone is not a speedup
   measurement. The installed general probe only permits devices 0/1, so use
   the repository's all-device inventory or `nvidia-smi` for the three-device
   question, without framework initialization.
3. If saved evidence does not independently check the score on the failed map,
   use a separately documented bounded value/score finite-difference diagnostic
   before assigning blame to the optimizer. Any such GPU diagnostic must use the
   archived q20 target, XLA, float64 and verified memory growth; at most 600
   supervisor seconds total (a convenience cost ceiling below the last recorded
   1,004.91-second diagnostic allowance). Verify the live budget first. No
   training ladder is authorized by this investigation plan.
4. Write a result note with source anchors, candidate tables, measured versus
   unresolved findings, and a practical repair order for three GPUs. Preserve
   unrelated working-tree changes and all previous experiment evidence.

Saved batch size, widths/depth, learning rate, clipping cap, training rungs,
validation size and stopping predicates are inherited hypotheses to audit,
not accepted production defaults. Report their actual values, mechanisms,
failure modes and missing calibration. Display windows and any activation
summary cutoffs are diagnostic convenience choices only. No new decision
threshold is inferred from them.

Skeptical pre-execution audit: the main risks are mistaking trial nomination for
optimization convergence, using a migrated counter as total effort, reading
current code as historical execution, treating clipping as a demonstrated cause,
ranking unequal-bank/unequal-update candidates, mistaking cached map parity for
independent target derivatives, and assuming three GPUs make one sequential
optimizer three times faster. The steps above explicitly address these risks.
This plan passes the audit for investigation; substantial retraining needs a
concrete follow-on allocation within the owner's existing campaign budget.

Outputs belong under
`docs/plans/artifacts/q20-training-gap-investigation-2026-09-22/` and in
`docs/plans/bayesfilter-q20-training-gap-results-2026-09-22.md`.

## Bounded derivative diagnostic specification

The saved qualification explicitly reuses the same analytic score and is not
an independent derivative check. Run one supervised diagnostic against scalar
central differences of the value, both in physical theta and latent z, using
the previously frozen map and exact same UKF approximate-posterior target.
Use 12 preserved points: eight spread through the saved iid bank and four with
the largest saved residuals. This convenience sample deliberately includes
stress cases; it estimates no population failure frequency. Evaluate all four
coordinate derivatives with relative increments 1e-3, 1e-4 and 1e-5 times
max(1,abs(coordinate)). Central differences have O(h^2) truncation and O(eps/h)
roundoff for a smooth value; this decreasing ladder checks stability near the
usual double-precision eps^(1/3) scale without assuming it is universally
optimal. Preserve every perturbed value/status and every derivative/error.

Use one stable 12-row compiled function per coordinate system, memory growth
verified before initialization, and no optimizer updates. The explanatory
agreement screen is scaled error <=1e-4 at both finer increments, where the
scale is max(1,abs(score),abs(finite difference)). This is an engineering
localization tolerance, not a posterior or universal gradient-accuracy
criterion. Nonfinite values or failed target statuses invalidate the affected
comparison. A persistent finite discrepancy triggers gradient repair before
further training; clean agreement weakens the gradient-bug hypothesis only at
the tested points. The maximum workload is 600 value/score rows plus setup,
bounded by the 600-second cumulative supervisor cap and the current campaign
diagnostic balance. Charge actual worker wall time to the existing campaign.

Pre-run audit: central differences consume values rather than the attached
custom gradient, so this can expose a faulty analytic score. Both spaces are
necessary to separate the physical target from the transport pullback. The
points and normalization are fixed, all coordinates are checked, and failed
rows are retained. It does not independently validate the UKF approximation
against the original nonlinear likelihood. This diagnostic passes the audit.

## Saved-data gradient decomposition for the clipping question

Explain which loss terms and network parameters produce gradients above the
inherited cap of 10 at the selected frozen map. Use the existing 1,000 iid
Gaussian points and transformed target scores, with the exact restored training
map from the r2 execution source. No target evaluations or optimizer updates.
Recover the physical score using the chain rule
`J.T * score_theta = score_z - grad_z(logdet)` and batched linear solves.
Check reconstructed physical positions and scores against the independent
12-point derivative diagnostic, and match the restored export to the saved map.
Use explicit coordinate VJPs, never pfor, to form the four-row map Jacobian.

On the first 992 points, take 31 consecutive disjoint batches of the actual
training size 32. Compute the exact frozen-point RKL parameter VJP and separate
negative log-likelihood, negative log-prior and negative log-determinant terms.
Record per-layer squared-norm shares, output scale versus shift contributions,
and variability across these batches. Compare against the saved training norm
history descriptively; these are new diagnostic batches at one fixed map, not
a reconstruction of past updates or a controlled optimizer comparison. The
remaining eight rows participate in score-reconstruction checks only.

All quantities are explanatory. Mismatched map/source identity, nonfinite
values, failed forward/score parity or failed additive gradient identities
invalidate the diagnostic. The inherited FP64 comparison tolerances are
rtol 1e-9, atol 1e-10. No clipping threshold, batch size, architecture or training
default is selected. The finite-bank mean and its variability do not establish
the population gradient, optimization convergence, whitening or posterior
accuracy. A large likelihood contribution can reflect genuine target
curvature, parameterization sensitivity and sampling variability; this
decomposition alone cannot assign a unique cause among those mechanisms.

Run `diagnose_saved_clipping.py` in the investigation artifact directory with
the tfgpu interpreter, but deliberately hide GPUs before framework import.
The pure map/derivative calculations use TensorFlow float64 with explicit
signatures and CPU XLA. This is a saved-data/reference diagnostic exception,
not CPU NeuTra training or CPU performance evidence. One supervised attempt
has a 120-second convenience ceiling, charged to the current diagnostic
allocation. Save inputs' checksums, source/command/environment, all batch
gradient components, comparison results and measured wall time in a fresh
campaign attempt, with a receipt in the investigation directory.

Skeptical audit: recovering physical scores is an algebraic reuse of already
checked scores, not another independent target-score test. The separate saved
physical evaluations and exact export match protect against a wrong-map
pullback. Fixed weights and disjoint batches distinguish observed batch
variability from changing parameters; neither gives an unclipped Adam history.
These restrictions make the diagnostic suitable for explaining clipping,
without turning it into an unsupported threshold recommendation. Audit passes.
