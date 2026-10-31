# Audit the implemented moment-training and guidance path

Status: COMPLETE. See
`artifacts/ledh-guidance-implementation-audit-20261020/result.md`.
23 focused CPU tests pass; consumer and result-assembly counterexamples confirm
two additional defects. Original544 rows pass the independent completeness
check. Diagnostic retention/source coverage and missing learner capabilities
are recorded separately. No runtime edits, new training, GPU charge or active
process. The next action is narrow diagnostic/evidence repair before another
training campaign, not additional numerical tuning in this audit.

Focused checks: run the existing physical-proposal, moment-training and SIR
guide tests with `CUDA_VISIBLE_DEVICES=-1`, preserving their diagnostic
non-XLA fixtures. Reproduce result-assembly acceptance using the existing
544-row holdout and copies with one missing replicate, one duplicate, and
one changed particle seed. These are integrity checks, not new stochastic
comparisons; original data and rankings are not modified. The pass criterion
is detection of the malformed copies while accepting the complete original.
The current inspection finds no algebraic inconsistency in the physical
mixture density or its parameter derivatives. Missing Jacobian and covariance
derivatives were investigated: the former has an explicit constant-Jacobian
contract, and the Gaussian fallback already includes covariance derivatives.
One further CPU consumer check uses the existing healthy two-dimensional
fixture with a deliberately inadmissible quadratic guide: negative guide
precision under the default PSD-only guide policy, but an SPD total proposal
precision. The guide-specific validity flags must be false, and the entire
program must reject the proposal. This separates a misleading diagnostic from
failure of the numerical rejection itself. No scientific ranking is involved.

20 October 2026. Owner question: are there additional wrong or incomplete
implementations beyond the completed training-failure diagnosis?
Worktree `ledh-iapf-execution-20260922`, branch `ledh-refactor-with-policy-fix`,
base `4d6915cc66`. Preserve all pre-existing changes. This is a review and
focused reproduction task, not authorization to change the scientific method.

## Question and scope

Trace the executed physical-proposal training campaign from joint simulation,
message fitting and adapters through sampling, mixture density, analytical
derivatives, UKF lifecycle, Contract E/GenUT, objective, candidate selection and
evaluation. Compare these operations to the mathematical document and actual
plan. Separate a wrong computation from an omitted capability, a deliberately
restricted diagnostic, a misleading completion claim and an untested concern.
Check adjacent reusable guide consumers when the same implementation claim
applies. Do not infer whole-repository correctness from this bounded audit.

Evidence contract: a confirmed defect needs exact source anchors and a
derivation or focused executable counterexample. A claim-bearing call chain
requires a consumer wiring/parity check; function existence is insufficient.
Previously checked objective/reset identities remain evidence only for their
checked scope. Existing heldout results are read-only; no new tuning, score
comparison or method ranking. Missing invariants, non-finite computations or
incorrect target/derivative semantics veto the affected correctness claim.
Runtime, small loss differences and message-fit quality are explanatory only.

## Procedure and skeptical review

1. Inventory the mathematical promises and active entry points; read the exact
   dependencies used by those entry points, including derivatives and guards.
2. Challenge conditioning, density/Jacobian consistency, guide/parameter
   dependence, covariance validity, simulated-label independence, optimizer
   selection, reported evidence and implementation coverage.
3. Reproduce material findings with the smallest algebraic or executable
   fixture. Use the shared endpoint where feasible and identify any scope a
   primitive check does not cover. Save complete logs with concise summaries.
4. Record severity, correct target, actual computation, effect, reproduction,
   repair requirement and uncertainty. Refresh the checkpoint and master.

Skeptical review PASS: this is not another efficacy campaign or a search for
arbitrary small numerical differences. Earlier descriptive findings do not
automatically become bugs. Frozen guide coefficients are distinct from guide
input derivatives; finite-program score checks are distinct from exact model
score claims. An explicit diagnostic restriction is not a false implementation
unless the documentation or consumer claims the omitted capability. Claims of
complete cSMC/KDM learning require the actual learner and downstream call chain.

## Defaults, resources and stop conditions

Start with source/standard-library inspection. Focused TensorFlow checks use
the existing `tftwogpu` environment; deliberately CPU-only fixtures set
`CUDA_VISIBLE_DEVICES=-1` before import and record a debugging/reference
exception. No eager or non-XLA diagnostic is production evidence. No new pfor,
NumPy runtime path, package installation or unreviewed numerical protection.

If a specific finding requires GPU parity, use escalated access and verified
memory growth, at most two GPU launches and 600 process seconds, charged to
the remaining 23826.197895057-second parent budget. Freeze the specific check
and numerical controls before launch. No broad sweep. Stop on a reproduced
defect rather than trying to tune around it; continue independent audit areas.
No change to defaults, heldout partitions, model scope or replicated algorithms.

Artifacts: `docs/plans/artifacts/ledh-guidance-implementation-audit-20261020/`.
Final `result.md` distinguishes checked defects, incomplete implementation,
correct checked behavior and not-checked scope. Pre-mortem: a test may verify
only a scalar fixture while missing particle-dependent or parameter-dependent
terms; inspect the complete call chain before reporting a pass. A mathematically
correct restricted pilot must not be relabeled as the complete proposal.
