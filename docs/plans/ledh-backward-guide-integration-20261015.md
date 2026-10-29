# Backward-information guidance inside the existing LEDH recursion

## Authorized task and scope

Document the proposed smoothing-informed guidance in the existing LaTeX
manuscript; audit with MathDevMCP; implement a proposal-only integration into
the shared LEDH executor; test the mathematics/code correspondence; audit again.
The user authorized execution on 2026-10-15 and subsequently requested a scoped
commit on surrogate-hmc, integration with main, and synchronization to the LEDH
branch. No additional agents, training campaign, scientific/default promotion,
or new push is included in this synchronization task.

The reader should be able to distinguish a backward information function from
ancestral smoothing, reconstruct the proposal correction, and differentiate
the same finite scalar through OT and Contract E. Preserve all existing
manuscript results and historical qualifications.

## Research intent and skeptical preflight

Question: can a frozen backward guide change the particle proposal without
changing the observation density, forking the canonical recursion, or losing
its analytical total derivative? Score accuracy improvement is a later question.
Candidate: a positive Gaussian/quadratic proposal guide, with a controlled-SMC
quadratic-policy adapter and an optional frozen Gaussian-mixture local guide.
The mixture is a representation of backward information, not an extra
observation likelihood and not an IWSG replacement for a total derivative.

Baseline: exactly the same shared executor with guidance disabled. The owner
setting epsilon=1 and Sinkhorn iterations=5 is an untuned development
comparator; retain and report balancing, ridge, correction and cap settings
separately. Do not overwrite any global defaults.

Skeptical audit: the previous two-cloud Fisher proposal does not meet the
finite-value/total-gradient requirement and will not be implemented as if it
did. Importance correction before deterministic reset does not prove an
unbiased full-reset filter. Frozen-policy partial differentiation must not be
called differentiation of an online refit. OT couplings are not backward
smoothing probabilities. Constant guidance must recover the baseline.

## Evidence contract and assumptions

Primary engineering pass: analytical directional derivatives agree with
central differences of the identical finite scalar on valid interior
fixtures; disabled/zero guidance reproduces baseline; tests exercise the
actual shared flow, weight, UKF, OT, Contract-E and correction consumer.
Hard vetoes: nonfinite output, invalid covariance/guide precision, missing
proposal correction, unsupported transition-density assumptions, incomplete
guide tangent, target substitution, or a divergent execution fork.
Explanatory only: runtime and one-fixture score differences. No superiority,
unbiased full likelihood, model-score consistency, tuning readiness or HMC
readiness follows from these tests.

Defaults audit: frozen guide coefficients are an explicit conditional
finite-program definition, not a model-specific trained default. Gaussian
transition support is an implementation restriction to check at entry. A
positive-curvature mixture local approximation is a proposal hypothesis, not
the exact Hessian or a published controlled-SMC algorithm. Its null/constant
limit and full input tangents are early checks. All fixture caps/ridges are
declared numerical choices, used for derivative checks only.

No stochastic ranking is planned. Future score-quality research must construct
and evaluate conditionally: bootstrap PF, current LEDH, current-observation
guidance, and exact Kalman/reference smoothing where available. Salient
conditions are uninformative futures, ambiguous current observations, strong
future information and poor guide fit. A heuristic loss vetoes promotion, not
the present engineering work.

## Stages and budget

1. Derive propositions in the manuscript, including assumptions, proofs,
   no-improvement counterexample and implementation-level pseudocode.
2. Run bounded MathDevMCP derivation audit; resolve material mathematical
   findings and review the plan against actual code signatures before edits.
3. Implement one optional proposal extension in the shared executor and
   reusable analytical guide primitives; do not duplicate the LEDH filter.
4. Run focused CPU-only debug/reference tests (GPU deliberately hidden),
   document build, wiring checks, and post-edit MathDevMCP consistency audit.
5. Save results, limitations and next justified action in the checkpoint.

Budget: at most 20 minutes CPU validation and three repair/test attempts; two
bounded audit rounds with targeted followups only; no GPU or model training.
Stop for a material failed mathematical assumption or exhausted budget, not
for a tool's lack of proof certification. Tool structural agreement is not
mathematical proof. Fresh logs preserve every attempt.

Artifacts: docs/plans/artifacts/ledh-backward-guide-20261015/; all commands,
commit, CPU mode, seeds, wall times and audit status will be recorded there.
## Preimplementation review

The manuscript now includes ledh_backward_guidance.tex in its main narrative.
The shared executor already separates physical density evaluation from flow
proposal geometry. The insertion point is its pre-flow draw; the corresponding
denominator must be replaced at the same time. Physical UKF prediction/update,
flow, weights, reset, and corrections remain shared, not copied.

Equation map: bg-gaussian and bg-gaussian-tangent -> guided draw helper;
bg-mixture and bg-mixture-tangent -> frozen mixture evaluator;
bg-weight -> existing executor's proposal denominator;
bg-total -> existing log-sum-exp and complete reset tangent recursion.
Use a separately named candidate kernel factory, not a new registered
canonical identity. It compiles an explicit fixed input_signature, defaults
to XLA, and requires the full reset/correction options. This first consumer
is single-cloud only; no claim of batch-lane or fitted-pilot integration.

MathDevMCP's tree audit failed internally; the bounded audit-and-fix workflow
examined four proposition labels successfully. It reports diagnostic-only,
no_proposal, no veto reasons, and unresolved proof obligations rather than
certification. Its concerns about inverse/logdet domains are covered by the
stated SPD/bijection assumptions and will receive executable checks. Compound
equations were expanded into single derivation lines. Formal certification
remains open. The subsequent LEDH-branch continuation checks the conditional
flow Jacobian with a derivation and actual-consumer finite differences; see
the [validation result](artifacts/ledh-backward-guide-jacobian-20261016/result.md).

Review verdict: proceed with bounded engineering checks; no scientific
promotion. No change to observation likelihood, categorical sampling, Fisher
score substitution or runtime policy refitting is authorized by this design.
Status: the original checks completed on surrogate-hmc. Integration against
main's newer shared executor passed 35 focused CPU reference tests on
2026-10-16. The merge and synchronization are complete; the LEDH checkout is
at `ea1018de1da55ed02a29a8d60f08a6827fce8a42`. See the historical integration
[result](artifacts/ledh-backward-guide-20261015/result.md) and
[run manifest](artifacts/ledh-backward-guide-20261015/run-manifest.json).

## Postimplementation review

The guide draw and its density denominator enter the same shared executor;
the physical observation density and existing UKF/flow/OT/Contract E/GenUT
recursion retain their roles. All ten focused tests pass, including normalized
proposal density, full covariance and mixture tangents, same-finite-program
directional derivatives, zero-guide parity and executable consumer wiring.
These CPU-only, non-JIT checks establish bounded engineering consistency, not
exact model-score correctness or GPU/XLA readiness.

The manuscript builds successfully and all five new PDF pages were visually
inspected. The final MathDevMCP audit examined all six propositions but remains
diagnostic-only, with unresolved formalization obligations. Its three lexical
document/code comparisons report mismatch; they are not semantic verification.
The explicit equation map and executable checks supply the scoped consistency
evidence. The later conditional-Jacobian validation adds an invertibility
proof and two nonlinear actual-consumer tests, with 37 regression tests passing
on the LEDH branch. Its label-scoped MathDevMCP audit remains inconclusive.
No score improvement, tuned default or canonical/HMC admission follows.

The [Jacobian continuation plan](ledh-backward-guide-jacobian-validation-20261016.md)
records the completed prerequisite and its limits. The subsequent user
continuation was executed under the separately audited
[scalar fitted-guide plan](ledh-fitted-guide-pilot-20261016.md). That bounded
GPU/XLA reference campaign is complete: 624 rows, all three conditions failing
the heuristic promotion check. Its result is in
`artifacts/ledh-fitted-guide-20261016/result.md`. The subsequent
[Gaussian baseline diagnostic](artifacts/ledh-gaussian-score-error-20261016/result.md)
stopped at 255/280 rows on a transport-validity veto, localized in CPU replay
to terminal column balancing. Finite-program derivative fixtures pass while
persistent model-score errors remain. Its checkpoint now governs the next
bounded balancing repair and first-step proposal/weighting investigation;
neither nonlinear transfer nor trained KDM is complete.

## Completed branch synchronization

Initial heads were surrogate-hmc `feddae400`, main `023247142`, and
ledh-refactor-with-policy-fix `b28c25a6c`. Main and LEDH were clean. The scoped
backward-guide implementation, tests, manuscript and evidence were committed
as `ad5de4a6d`, integrated into main, and synchronized to LEDH at
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`. Unrelated surrogate edits were
preserved. This synchronization is finished; it is not the next action.

Pre-execution skeptical audit: copying the old executor would discard main's newer ancestry,
affine-flow, and marginal-mixture weight implementation. Preserve those paths
and integrate the guide at the shared draw and proposal-density call chain.
Reject combinations whose corrected proposal density is not implemented.
Repeat focused engineering tests in the integrated checkout; the original CPU
results do not validate the newer executor. The comparator remains the same
executor without guidance, with no score ranking or default promotion.

The completed integration allowed up to 20 additional minutes of CPU-only
validation and three localized repair attempts. Stop conditions were unresolved
proposal-density semantics or destructive overlap with unrelated edits.
Conflict decisions, checks and heads are recorded in the integration evidence.
No research campaign was added.

Integration check: surrogate commit `ad5de4a6d` was reconciled with main
`023247142`. The newer executor paths and marginal-balance validity guards
are retained. Two missing tracked sparse-checkout dependencies were restored.
The guide reference fixture needed 32 terminal balancing iterations to satisfy
main's existing column-residual guard; epsilon=1 and five Sinkhorn iterations
remain unchanged. This fixture repair is neither a runtime default nor tuning
evidence. The final suite passed 35 tests in 128.82 wall seconds. Test and trace
commands used 290.92 seconds of the additional budget, with three localized
repairs. The evidence supports engineering integration only; the open scientific
questions and promotion restrictions above remain in force.

## Follow-up scalar pilot

The next target-specific stage was executed separately under
`docs/plans/ledh-fitted-guide-pilot-20261016.md` on the LEDH checkout. It
integrated the actual replicated scalar controlled-SMC fit, exported the
future-only quadratic factor, and passed that factor through the unchanged
full analytical LEDH consumer. The GPU/XLA Gaussian reference completed with
624 finite rows, but every condition retained a heuristic promotion veto;
these results are exploratory and do not promote guidance, tuning settings,
KDM, HMC, or nonlinear performance. The complete decision and inference
status are in `docs/plans/artifacts/ledh-fitted-guide-20261016/result.md`.
