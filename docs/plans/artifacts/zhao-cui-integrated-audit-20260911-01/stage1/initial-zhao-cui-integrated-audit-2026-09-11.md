# Integrated Zhao-Cui mathematical and implementation audit

Owner request: consolidate the context protocol, then thoroughly analyze current
code and LaTeX, propose coherent fixes, and prove mathematical assertions with
MathDevMCP. This supersedes the single-guard next-stage brief for this turn.

Policy consolidation is complete: shared source plus nine installed copies,
obsolete Codex-only block removed, unrelated text preserved, and installer
idempotency checked. Evidence: artifacts/context-policy-install-20260911-01/.

Research checkout: /home/chakwong/BayesFilterZhaoCui, base
47176bce7cdaa91ddd4466c39f90a11ad005c800 plus saved uncommitted work. Runtime
repairs are proposals in this task; diagnostic scripts and mathematical proof
artifacts may be written in the BayesFilter notes workspace. Preserve all dirty
code and ignored manuscript files. The active mathematical target is the
Algorithm 3 section of attempt05_n4_failure_analysis.tex and its dependencies,
not the retired APF campaign.

## Audit contract and skeptical preflight

Question: do the document, proposal preparation, sampling law, finite value,
analytical score, and consumer endpoints define and compute one coherent
algorithm? Check against the local Zhao-Cui paper Algorithms 2-3, equations
(13)-(23), and author marginal/conditional code. Distinguish source operations,
fixed-parameter adaptations, and extensions. No stochastic superiority or
filtering quality claim follows from exact identities or short tests.

Primary criterion: explicit same-target derivations and bounded independent
executable/proof evidence. Vetoes: wrong law, wrong derivative, accepted invalid
result, unsupported mathematical equivalence, and absent consumer wiring.
Explanatory: observed numerical error, fit residual, runtime, and ESS. Candidate
defects trigger repair proposals; they do not invalidate the research direction.
Continuation veto: unavailable required source, unresolvable environment problem,
or exhausted bounded diagnostic budget; document remaining proof gaps honestly.

Skeptical preflight passes for an audit, not an experiment. Avoid conflating
fixed sampled states with fixed uniforms, smooth TT with an interpolated CDF,
positive support with finite importance-weight variance, rank capacity with ALS
activation, or isolated function availability with end-to-end implementation.
Finite algebra certificates cannot certify sampling, floating-point safety, or
optimization convergence. Tool proof statuses must be inspected, not inferred
from successful API calls or a structural audit.

## Stages and budget

1. Inventory current source, LaTeX propositions, previous counterexamples, and
   source hashes. Trace consumers and distinguish production graph claims.
2. Derive the complete mathematical algorithm and expose every inconsistent
   target or omitted hypothesis. Use MathDevMCP symbolic/Lean backends for
   mathematical claims. Save exact inputs and returned evidence.
3. Run only discriminating deterministic CPU/reference diagnostics needed for
   new concerns. Reuse unchanged saved evidence. Up to six numerical invocations
   at 180 seconds each, including repairs; no GPU, HMC, training, tuning, or
   serious comparison. Hide GPU before framework import. No environment installs.
4. Up to twelve bounded symbolic calls and eight Lean attempts of at most 60
   seconds; adjust only for a documented infrastructure-only retry under the same
   allowance. Write local proofs and explanations before seeking tool verdicts.
5. Deliver one coherent repair proposal with priorities, mathematical rationale,
   affected call chains, regression requirements, and explicit proof status.

New artifacts: docs/plans/artifacts/zhao-cui-integrated-audit-20260911-01/.
Record commands, backend/environment, source hashes, elapsed time, failures,
and proof assumptions there. Float64/interior polynomial fixtures are diagnostic
choices, not scientific defaults. Existing counterexamples use NumPy/SciPy only
as independent references. Keep outputs bounded and save progress after each
stage; do not reload prior conversations.
