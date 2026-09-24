# Phase 0A: skeptical audit and execution contract

Authorization: the owner's 2026-09-14 request to thoroughly review and execute
the master, including implementation and phase-boundary repair. The master is
`docs/plans/younis-kdm-score-master-program-2026-09-14.md`; this note records
its execution, not an additional prerequisite program.

## Audit decision

Proceed with implementation. The master now separates finite-program
derivatives, gradients of mixture expectations, likelihood/derivative pairs,
and model scores; it requires appropriate evidence for each. It distinguishes
candidate rejection from a continuation veto and includes bounded repairs,
scope-specific tuning, independent holdouts, paired uncertainty, heuristic
comparators, and total cost. No empirical superiority or default promotion is
justified by this audit.

The audit checked wrong baselines, proxy criteria, missing stops, unfair
comparisons, hidden defaults, stale context, environment mismatches, and
whether the commands would answer the question. Corrections made before
implementation:

| Finding | Disposition |
|---|---|
| Manuscript used C4 to justify fourth-order finite differences. | Corrected to C5 with bounded fifth derivative, a counterexample, explicit stencil moments and proof. |
| Manuscript required truncation slopes in stochastic particle error curves. | Separated exact deterministic order from full stochastic MSE, retaining the bias cross-term and coupling covariance. |
| SGQF text omitted prediction/conditioning/carry/reset dependencies and treated normalized integration weights ambiguously. | Added the additive-noise moment recursion, observation conditioning, signed-weight distinction, reset and sensitivity requirements. |
| One iAPF sentence incorrectly reduced its role to a UKF replacement. | Clarified auxiliary ancestor and state proposal changes and their derivatives/corrections. |
| UKF on Gaussian models can equal the exact oracle. | Explicit correctness-only interpretation; a particle method need not beat an exact solution before the planned nonlinear experiment. |
| Canonical executor and KDM integration have an incompatible callback/trace contract. | Verified statically; a real endpoint regression is required before either KDM row is eligible. Repair belongs to 0D. |
| Gaussian reference differentiates A only. | Broader parameter claims blocked until 0C implements and tests initial mean/covariance, Q, R, H and A dependencies. |
| Tuning scope/registry types exist, but an eligible study issuer/consumer has not been located. | Treat as implementation work in 0C. A matching caller-supplied label is insufficient. |

The checkout changed concurrently from the starting revision to
`e7f2a88ecff49ced481b9615c0b1237b8cabe732` during source inspection. Preserve
those canonical loop repairs; snapshot actual source closures in each run and
invalidate affected results on resume. Historical speed claims are not reused.

## Research intent and return contract

Main question: can KDM/IWSG, better proposal moments, corrected twisting, or
coupled finite differences reduce model-score MSE at a declared budget?
The expected failure is that a differentiable or low-variance estimator targets
the wrong quantity, or retains enough normalization bias to lose to a simpler
method. The initial Gaussian work tests identities and orchestration, not
scientific superiority.

Every numerical return identifies: model; physical likelihood; computed value
target; computed derivative target; comparison target; parameter coordinates;
full initial-law dependence; actual proposal and denominator; particle/data
streams; numerical settings; timing; engineering and numerical validity; and
scientific inference status. Comparing a finite-program derivative with the
model score is an explicitly labeled approximation-error evaluation, never an
assertion that their targets are equal. IWSG expectation gradients get a
separate target identity. Autodiff is permitted for diagnostic parity only.

| Diagnostic | Role |
|---|---|
| Same-scalar analytical/FD identity, sampler/density normalization, initial-law dependence | Engineering/numerical veto; repair the producer before continuing that row. |
| Support, finite values, covariance/domain validity, intact results | Continuation veto for the affected row; independent valid branches continue. |
| Untouched oracle score MSE and predeclared paired uncertainty at equal budget | Primary scientific criterion in a later adequately powered study. |
| Significant loss to a constructed cheap comparator in a salient regime | Promotion veto, not automatic abandonment of the research direction. |
| ESS, covariance error, weight tails, training fit, short-run means | Explanatory or nomination only; no superiority inference. |
| Changed source/specification/partition/tuning | Invalidate reuse; rerun or repair in fresh attempt directories. |

## Defaults and initial allocation

Use the existing `tftwogpu` environment, TensorFlow 2.20.0.dev0+selfbuilt,
TFP 0.25.0, pytest 9.0.2. Python:
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.
GPU/XLA is the candidate target; explicit CPU/FP64 references hide GPU before
import. Stable TensorFlow signatures and analytical recursion are required.
No NumPy runtime, pfor, or package/environment mutation is planned.

| Choice | Provenance, risk, early diagnostic and status |
|---|---|
| Small Gaussian models first | Master oracle ladder; exact adaptation may hide nonlinear failure. All-parameter derivatives and independent FD first; correctness fixture only. |
| Tiny initial N/T and fixed streams | Resource/debug convenience, not tuned scientific settings. Verify row shape, source-to-consumer wiring and stream replay; no ranking. |
| Existing canonical settings | Warm starts only. Check numerical margins and same-scalar parity; actual scope tuning required before a claim. |
| Positive FD ladder | Master derivation; too small h causes cancellation, too large h truncation/domain error. Test node distinctness, domains, deterministic slopes and measured covariance. |
| Initial 12 CPU process-hours / 8 GPU device-hours / 12 GPU launches | Master validation allocation, including failures/compilation; reserve 2 GPU hours for repairs. Refresh later finite allocations from measured cost. |

Stop an affected run on invalid target/support, unrepaired identity failure,
corrupt evidence, unavailable required diagnostics, or budget exhaustion. A
missing method is a visible blocked row. Three failed repairs of the same
cause require reassessment. Neither reviewer unavailability nor a failed
candidate creates an extra owner-approval requirement.

## Evidence and next phase

Protected inputs and initial provenance are in `baseline/` and
`starting-provenance.json`. Four MathDevMCP/SymPy checks establish the scalar
algebra for the MSE cross-term, both fifth-order stencil moments, and the
quadratic-MSE optimal coefficient. They do not certify interchange of
derivative and expectation, the entire manuscript, or runtime correctness.
The separate bounded manuscript audit and document build are retained here.

Next: implement 0B's registries, target validation, versioned attempts, bounded
repair/resume, and reporting; test them with explicitly synthetic endpoints.
Then replace the synthetic fixture with 0C's actual TensorFlow baseline
adapters. Missing KDM/SGQF/twist endpoints remain blocked until their producer
phase and call-chain tests pass.
