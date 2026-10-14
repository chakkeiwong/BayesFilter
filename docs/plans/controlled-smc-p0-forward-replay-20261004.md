# P0: matched author forward-consumer replay

Authority: the replication master and the owner's direction to reproduce the
author's behavior, including intrinsic failures, before considering improvements.

This phase tests the actual TensorFlow forward consumer on the author's fixed
policies and original random draws. Fitting parity and repeated-run no-worse
qualification remain separate unfinished work. Paper-figure success is not a
dependency.

## Evidence contract and research intent

Question: with the same policies, Gaussian draws and systematic-resampling
uniforms, does the existing TensorFlow forward endpoint reproduce the six
audited author runs (22 stages), and reject the saved intrinsic invalid policy?
The comparator is pinned Heng author code, commit
2696f6d611aae172d10102cb7bfbfd314bb6b88f, under the recorded Octave runtime.
Use the original six cases from attempt86: forcing 2.5, 4.8801 and 8.5, seeds
20260929 and 20260930, N=512, d=8, T=100, adaptive maximum I=4.

The CPU capture executes unchanged and observation-only copies. Both must
exactly reproduce every original returned field; the observed copy must also
end with the unchanged copy's rand/randn states. Capture the actual randn
matrix in mvnrnd and the actual uniform in Systematic_Resampling, preserving
their call order and arithmetic. Copying already-produced trajectories into
the TensorFlow output or reconstructing noise from them would make this test
circular and is forbidden.

The GPU phase calls MultivariateGaussianControlledSMC.forward, not a replacement
forward implementation, for every stage. It records trajectory, ancestry,
incremental weights, cumulative log normalizer, ESS, and numerical validity.
Use the existing diagnostic value screen atol=1e-7, rtol=1e-9 and exact ancestry
comparison. These are localization screens, not calibrated P1 equivalence
margins. Report the first divergent time and all invalid outcomes. A mismatch
triggers source-preserving localization; it does not authorize author repair.
Do not rerun the old solver search or rank solvers by residuals.

The saved failure replay must export the actual failed policy and covariance
inputs. Check the failed precision through the shared TensorFlow covariance
routine and the full forward endpoint's invalid-result flag. This is a policy
domain check; synthetic draws used for this invalid policy do not establish
matched failing-trajectory replay. An intrinsic source failure is an accepted
reference outcome. A TensorFlow success flag on an invalid policy, or an extra
invalid result on a valid source stage, vetoes this limited forward claim.

Wrong source/data, altered unchanged outputs/RNG, corrupt fixtures, missing
random draws, incorrect device, missing growth policy, or a missing consumer
endpoint invalidate the experiment and stop dependent execution. Diagnostic
execution completion and numerical comparison verdicts are distinct. A
completed diagnostic may report mismatches; no whole-replication admission is
issued. Condition numbers, timings and error magnitudes explain results only.
No stochastic ranking, fitting parity, full Figure6 replication, MATLAB-runtime
equivalence, KDM completion, or default-readiness claim follows from this phase.

## Source and assumption audit

Read source anchors: published paper §3.1 equations (11)-(13), §3.2 Algorithm 1,
and §6.2 equations (42)-(44); Lorenz/cSMC_Resample.m:51-140 (sampling, weighting,
ancestry), Systematic_Resampling.m:10-19, RungeKutta4.m; Octave statistics
inst/distributions/mvnrnd.m:23,66-85. The latter uses
chol(Sigma + eps*norm(Sigma,'fro')*I), with its documented eigen fallback.
The current TensorFlow sampler uses Cholesky without this runtime perturbation.
Preserve and report that known runtime difference; do not silently change either
sampler in this measurement. Chaotic propagation or resampling-boundary changes
can amplify roundoff, so failed pathwise screens are not automatically failures
of stochastic equivalence.

| Choice | Provenance and reason | Risk and earliest check | Status |
|---|---|---|---|
| Six cases / 22 stages | Existing exact source capture | Limited support; enumerate all stages, retain full-run gaps | Fixed diagnostic cases |
| Float64, TF32 off, GPU/XLA | Existing port comparison | Device/compile differences; record placement and single traces | Reference comparison |
| Source normal-equations setting; precision_policy=none | Author full-quadratic route | Indefinite policies; preserve invalid flag and failed fixture | Frozen baseline, no fit performed here |
| Native Octave random stream | Actual source-generated draws | Logging changes stream; exact output and end-state comparison | Required comparator |
| Existing tolerance | Prior localization only | Not an equivalence margin; separate screen from P1 acceptance | Diagnostic hypothesis |
| No added ridge, clipping or floor | Owner's replication-only target | Existing source failure remains possible and accepted | Owner-directed baseline |

Heuristic adversaries: not applicable to fixed-input code equivalence; no method
efficacy comparison or heuristic-quality promotion is made. Conditional coverage
is all three forcings, both seeds, all stages, and the intrinsic failed policy.

## Execution, budget and review

Keep the active source job running. Serialize the following after its terminal
state: the already queued 60-second failure replay, then p0_forward_capture
(CPU, 90 seconds), then p0_forward_compare (GPU, 600 seconds). At most one attempt
per new phase. These use the existing CPU reserve and GPU allocation, with no
new compute grant. Raise the aggregate attempt ceiling from 147 to 148 solely
to accommodate this two-phase continuation after attempt146. No retry or cap
extension is implicit; localized faults must be recorded and reassessed within
the unchanged compute allocation.

Use the existing absolute Python/dispatcher command form with start --phase
p0_forward_capture and start --phase p0_forward_compare. Every attempt gets its
own versioned output directory, source snapshots, input hashes, command,
environment, random seeds, hardware/memory policy and wall-time manifest.
The CPU capture intentionally hides CUDA. The GPU command requires trusted
execution and verified memory growth before device initialization.

Skeptical pre-execution review: PASS for this limited comparison. It removes
the missing-common-draw limitation of attempt90, invokes the real consumer,
preserves author failures, and makes no solver or robustness change. The
known Octave covariance perturbation and chaotic/resampling sensitivity limit
pathwise interpretation; they are recorded rather than hidden by a tolerance
change. A passing command is not full replication. Focused software tests must
verify source instrumentation, queue dependencies, source snapshot wiring and
separation of diagnostic completion from comparison acceptance before launch.

## Software review evidence

Before queue activation, the skeptical implementation audit found and corrected
two harness issues: the author source current directory could shadow the
instrumented resampler, and the comparison expected flattened snapshots although
the dispatcher preserves dependency subdirectories. The capture now leaves the
source directory after its original setup/data load and checks resolved paths.
The GPU loader reads the dispatcher's actual snapshot paths.

Focused checks: 32 tests and five subtests PASS, including a real Octave
comparison of original/instrumented sample values, resampling ancestors and final
RNG states; budget reservation for running and pending phases; actual comparison
command/input construction; and rejection of missing inputs or false acceptance.
All six archived cases were inspected for the expected 101-step policy/cloud
shapes and their 5+5+2+2+4+4=22 stages.
Evidence: artifacts/controlled-smc-kdm-replication-20260928/p0-forward-review-20261004-01/checks01.json
and checks01.log. These are harness checks; GPU consumer parity awaits the
serialized numerical comparison.
