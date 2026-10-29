# Supervised cSMC/KDM guidance for analytical LEDH scores

Completed: all seven bounded implementation steps, including the fresh T5
pilot, terminal audit and manuscript update. See
[result.md](artifacts/ledh-supervised-guidance-20261019/result.md).
Thirty-four distinct tests pass; T5 has 96 valid rows and six passing
directional checks. No statistical method ranking or guide promotion.
Stage GPU charge 570.83 seconds; campaign remaining 33303.439085210 seconds.
The preserved pre-run contract below explains what executed. Further
scope-specific calibration and independent-dataset replication are new
research allocations, not silently covered by these pilot results.

Owner instruction: document and audit the proposal, build the PDF, review the
implementation plan, refresh both masters, and execute. Work only on
`ledh-refactor-with-policy-fix` in `ledh-iapf-execution-20260922`. Preserve the
existing dirty tree. No commit, push, environment mutation or OT tuning.

## Research question and evidence contract

Can simulation-supervised selection of cSMC/KDM guide settings reduce the
mean squared error of the analytical score produced by the shared full LEDH
executor? Contract E and OT epsilon=1, Sinkhorn iterations=5 remain fixed.
Replication is closed. New fitting objectives are local method extensions.

The mathematical result is a population squared-risk identity, not a claim
that finite training, moment matching or lower message-regression loss improves
the score on the two previously evaluated SIR observation sequences. Simulate
fresh observations and latent paths from exactly the consumer's model. Keep
the latent labels inaccessible to guide fitting and particle execution.

Baseline ladder: unguided full LEDH; the original cSMC quadratic export;
original positive-information kernel consumption; supervised cSMC settings;
supervised kernel settings with exact Gaussian-product mixture moments.
Cheap adversaries are zero guide, half-strength quadratic guide and the
unchanged quadratic guide. Their rationale is respectively avoiding erroneous
future information, moderating it, and retaining the original fit. Evaluate
separately by observation horizon and, if sample size permits, low/high latent
label magnitude. Heuristics are falsification comparisons, never hidden
training labels. Selection among guide settings uses training/validation only.

Primary numerical pass: correct risk decomposition, exact-law complete-data
labels, correct analytical score tangents and actual endpoint-to-shared-executor
wiring. Primary scientific criterion on an adequately replicated untouched
set: negative paired mean difference in complete-data-score squared loss,
with a paired interval excluding zero and no componentwise heuristic veto.
Few-dataset pilot differences are descriptive only. Moment loss, backward
regression residual, ESS, runtime and precision margins explain results;
none independently establishes score improvement. Wrong model/timing, leaked
latent labels, non-finite scores, changed OT, incorrect derivatives, missing
full correction, or artifact corruption veto the affected comparison.
Invalid candidate proposals reject that candidate, not the research direction.

## Mathematical and implementation audit before execution

The reader knows particle filtering and differentiation but should be able to
reconstruct conditional-risk orthogonality, Fisher's identity, the training
and evaluation split, and how each guide changes the actual LEDH proposal.
Prose remains provisional pending reader feedback; mathematical checks are
independent of that feedback. Preserve the existing manuscript and PDF before
adding one self-contained section. No old equations or findings are removed.

Skeptical audit findings and resolutions:

* The earlier campaign's three-density consumer is not the shared full LEDH
  algorithm. New score comparisons must call `build_backward_guided_ledh_kernel`
  and its shared analytical executor, with UKF covariance propagation and full
  GenUT correction. Do not relabel previous comparisons.
* The active SIR scope uses the continuous Gaussian transition in
  `austria_sir_canonical_model`; it is not the clipped author-simulator law.
  Generate labels from that exact consumer model. A clipped-law teacher would
  answer the wrong question. The document treats clipped models separately.
* The score identity requires fixed latent paths when taking parameter
  derivatives, independent particle noise, fixed loss scaling, and sufficient
  integrability. Prove these assumptions explicitly. Training-score evaluation
  never differentiates through the simulator noise to obtain the label.
* A guide fit at the evaluation parameter is frozen while the analytical score
  is computed. No derivative through a theta-dependent refitting operation is
  claimed. A learned theta-dependent guide would need its total derivative.
* Gaussian mixture moment matching preserves product moments, not the mixture
  density. Use the Gaussian actually sampled in the importance denominator.
* Original KDM NLL and optional mean-MSE training are source results; feature
  and complete-score supervision of LEDH are adaptations, not replicated claims.

With these corrections the implementation plan passes this pre-execution
audit. Independent mathematical/tool review and executable checks follow.

## Ordered execution

1. Add propositions/proofs and explicit algorithm to the existing LaTeX
   manuscript. Audit with MathDevMCP, direct algebra, source anchors and rendered
   PDF inspection. Report tool limitations rather than calling them proof.
2. Implement TensorFlow supervised moment/score losses and exact-law SIR
   complete-data labels using the existing model callbacks. Add a paired-risk
   report and partition checks. No NumPy runtime or pfor.
3. Wire optional product-moment consumption into the shared guided draw and
   analytical tangent. Preserve current default. Permit indefinite quadratic
   information only through the existing positive-total-precision check; retain
   raw fitted coefficients for this candidate rather than disabling safeguards.
4. Verify risk decomposition against an exact conditional Gaussian example;
   complete-data labels against fixed-path finite differences; mixture moments
   and density tangents; no-fire default parity; actual full-executor wiring.
5. Execute a fresh GPU/XLA SIR supervision pilot. First use a short horizon to
   expose runtime, derivative or covariance failures. Fit cSMC/RBF guides using
   observations only; select frozen strength/consumer settings using separate
   supervised calibration and validation sequences; evaluate untouched data.
   This first implementation selects a bounded bank of guide settings. Neural
   capacity training and differentiating a score loss through guide coefficients
   are separate extensions, not silently claimed as implemented.
6. If the pilot is valid, use measured runtime and remaining budget to choose
   the next SIR horizon/replication rung; write its exact allocation before
   launch. If a real shared-executor failure occurs, localize and repair it
   within scope rather than falling back to the old diagnostic filter.
7. Update the manuscript implementation/evidence paragraph, both masters and
   concise checkpoint with results, remaining work and budget. Build the final
   PDF and re-audit math-to-code consistency.

## Defaults, limits and resource accounting

### Measured first rung and next allocation

Pilot-02 completed T2/N64 in 261.32 process seconds, 96 rows. All five
non-raw arms were numerically valid. The raw arm failed because the old
FrozenQuadraticBackwardGuide still demanded positive guide information, even
though the documented proposal requires only positive total precision. This
is an implementation mismatch, not a failed scientific candidate. Repair it
through an explicit optional flag, preserve the old default, and test both an
admissible indefinite guide and an inadmissible total precision.

Pilot-03 uses fresh seeds offset by 10,000, T5/N64, the same six predeclared
arms and all other controls unchanged. Refit and select separately for T5; no
T2 settings transfer. Allocate 1,400 process seconds (hard timeout 1,450) and
retain all prior evidence. Total stage charge before this launch is 321.32
seconds, including the conservative 60-second initialization-failure charge.
This is the third launch within the original 3,600-second stage allocation.
The final accepted rung closes this bounded implementation pilot; longer
horizons, parameter coverage and confirmatory replication remain explicitly
separate research work, not silently satisfied by this mechanics result.

### Frozen first pilot and execution review

Run `python -m bayesfilter.score_study.supervised_guidance_pilot --horizon 2
--budget-seconds 1800 --output <unique-pilot-root>` in tftwogpu with escalated
GPU access, memory growth, and XLA command buffers disabled as in the preceding
campaign. Use eight independent observation seeds, partitioned 2/2/4; two
independent particle seeds per sequence; N=64; theta=(0,0,0). This rung tests
mechanics, not parameter-region coverage or statistical improvement. Freeze
validation-selected settings before entering the test partition.

Six arms: zero, half-strength and unchanged projected cSMC, raw cSMC, kernel
positive-information, and kernel product moments. Half-strength is both an
explicit bank member and a reported simple comparator; it is not an independent
post-selection falsification test. The untouched test comparison is descriptive
only. A future promotion run needs a separately frozen adversary set and
adequate independent datasets. This resolves the ambiguity in the initial
heuristic wording above without claiming this pilot clears that gate.

Full-executor controls are fixed: 24 flow steps; balance 32 after owner-frozen
OT (1,5); ridge 1e-5 and denominator regularization 1e-7; one diagonal and one
pairwise correction with strengths .2/.02; LM .01, scale floor .0001, trust
radius .5, pairwise RMS cap 2, coordinate cap .94 and power 8. These are
mechanics hypotheses from the shared implementation, not SIR tuning evidence.
Positive damping and bounded standardized updates control local correction
steps; ridge and denominator floors protect factorization and divisions. Their
sizes remain uncalibrated risks, so this pilot cannot select them or support
default readiness. The earliest checks are stage-validity flags and independent
finite differences on every first-dataset candidate. `moment_safety=False`
retains the comparator's higher-moment route; it does not remove the shared
domain/factorization checks. Assess any protection change in a separate
non-harm calibration rather than confusing it with guidance benefit.

Scale each score coordinate by max(1, calibration RMS label); the unit floor
prevents noisy two-dataset near-zero scales from dominating. This is a declared
pilot weighting convention, not a universal scientific default. Report all raw
scores and losses. No latent labels enter the guide fitter or filter.

Execution audit: full-executor call-chain parity and all 20 focused tests pass.
The batched observation label initially misused a single-observation callback;
it now uses the explicitly derived diagonal-Gaussian density and analytical
score, with fixed-path finite-difference and observation-indexing tests.
MathDevMCP's document audit fails internally in read_code; Claude review cannot
authenticate. These tool failures are recorded, not counted as approval or
mathematical disproof. Direct proofs and executable checks permit this bounded
diagnostic stage under the current review-proportionality policy.

OT (1,5) is owner-frozen for comparability, not claimed optimal. FP64/GPU/XLA
with TF32 off is a diagnostic numerical exception, not the production default.
Memory growth is verified before device initialization. Short-horizon N64 is
a wiring/feasibility rung; it does not transfer tuning to T50. K=N obeys the
chunk policy. The original 512-particle quadratic fit is a warm start; its
regression condition and total precision are reported. Kernel component count
and initial bandwidth grid are warm starts, never universal defaults.

Loss feature scaling is fitted only on calibration labels and then frozen;
score components are also reported unscaled. Equal scales are used in exact
unit checks. Squared-label noise may dominate small comparisons: paired
candidate differences share labels and particle seeds, and uncertainty is
computed over independent observation sequences, not particle replicates.
Calibration/validation/test seeds are disjoint and stored in manifests.

Total authorized campaign ceiling: 43,200 GPU-process seconds. Prior spending:
9,325.730914790 seconds; available 33,874.269085210. Initial new allocation:
3,600 GPU-process seconds including compilation and local repairs, 600 CPU
diagnostic seconds. Up to three versioned launches in this allocation; further
allocation can use the remaining owner budget after recording measured cost.
All launches use unique roots below
`docs/plans/artifacts/ledh-supervised-guidance-20261019/`.
Record command, commit, dirty dependency hashes, model identity, seeds,
environment/device/memory policy, complete controls, wall time and exact
outputs. No claim-bearing tuning artifact or canonical admission is issued by
this research pilot. Stop at budget, repeated invalid target assumptions, or
unrepairable numerical invalidity; do not promote a failed candidate.

Pre-mortem: a successful run could still learn the simulator's data distribution
poorly, overfit noisy labels, exploit an incorrect derivative, or lose to a
simple guide. Exact Gaussian identity checks, fixed-path/finite-program
derivative checks, separated data and conditional baseline tables distinguish
these failures. More capacity is not the automatic next response.
