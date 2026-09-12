# Claude handoff: HMC candidate-set unification R4 audit

Date: 2026-09-12
Review target: read-only audit of the R4 plan and bounded implementation
changes for BayesFilter's HMC candidate-set unification.

## Review instructions

Start with the smallest exact paths needed for each question. Do not edit files,
run tests, launch agents, inspect unrelated repositories, or treat the current
dirty worktree as a clean commit. The main plan is
`docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`.
The implementation paths are
`bayesfilter/inference/hmc_candidate_set_tuning.py` and
`bayesfilter/inference/hmc_candidate_set_artifacts.py`; the focused tests are
`tests/test_hmc_candidate_set_tuning.py` and
`tests/test_hmc_candidate_set_artifacts.py`. The guide/reference check is
`tests/test_hmc_tuning_documentation_contract.py` and
`docs/reference/hmc-tuning-interface.md`.

The R4 execution note and result are under
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/execution-r4-20260912T125338Z/`.
The R3 implementation root and earlier Claude reviews remain historical
context. Inspect only cited paths unless a source line explicitly requires a
next exact path.

## Background and intended procedure

The owner wants one sensible HMC tuning procedure: prepare one immutable
target/geometry scope, explore a broad set of leapfrog counts `L`, qualify
epsilon independently for every `L`, retain every non-vetoed candidate, carry
all survivors through declared stages and fresh verification, and return a set
with an optional descriptive nominee. Ordinary coordinates and frozen
transports have different target-preparation and mass rules, but their search,
retention, repair, verification, budget, and replay semantics should be shared.

The motivating MacroFinance failure was specific. A candidate at `L=3` became
inconclusive, a later attempt selected `L=5`, a higher-epsilon repair for that
candidate was computed, and the repair was never verified before generic budget
exhaustion. The repair must preserve exact `L`, mass, target, coordinates,
start-bank design, and warmup protocol; changing epsilon creates a new child
record in the same fixed-`L` family. A child receives fresh evidence and is
scheduled after already reserved work but before not-yet-admitted different-`L`
work. Inconclusive evidence cannot silently switch `L`, and a computed child is
not a handoff until its own fresh verification passes.

R3 implemented the pure state machine, immutable candidate/family/receipt
records, directional repair queue, typed statuses, atomic checksummed result,
and deterministic synthetic traces. R4 found four concrete gaps while auditing
that implementation:

1. A checksummed result did not itself prove candidate hashes, receipt identity,
   parent/child scope identity, or qualified repair status, and shared-invalidity
   evidence could still be replayed.
2. Resume worked only when the original controller object remained alive; a
   persisted partial result did not reconstruct work order, reservations, or
   verification-attempt ordinals.
3. The reference guide still presented
   `select_fixed_transport_candidate_set` as a second recommended validation
   lifecycle, contradicting the plan's diagnostic-only disposition.
4. Work items were logged as reserved but not charged at dispatch, so remaining
   budget was overstated and an unfunded exploration could appear complete.

R4 repairs those four boundaries. Scope/candidate identity now includes
backend, dtype, execution mode, adapter signature, and source-dependency hash.
Dispatch charges each work item once from its candidate reservation and terminal
release returns only unspent reserve. The result exposes used, reserved, and
remaining units. `from_result_payload` plus `resume_hmc_candidate_set` rebuilds
the checked state without recreating IDs or rerunning completed work. Artifact
validation checks candidate-record hashes, exact `L`/epsilon/mass, all parent
and child scope fields, unique verification attempts, non-overlapping credited
draw ranges per stream, directional ordering, status precedence, and the
shared-invalidity replay veto. The reference guide now describes per-scope
results and read-only cross-scope reporting; the legacy selector example is
explicitly diagnostic-only.

## Questions for this audit

1. Is the R4 budget model coherent? In particular, does charging work from a
   held candidate reservation and returning only unspent units conserve total
   budget across interruption, terminal release, repair-child allocation, and
   persisted reconstruction? Are `repair_budget_exhausted`, `partial_budget`,
   and `paused_infrastructure` still distinct and truthful?
2. Does `from_result_payload` preserve the exact active-cohort order, deferred
   repair order, candidate IDs/hashes, verification-attempt IDs, and reserve
   ledger on every interruption boundary? Identify any payload state that is
   needed for a deterministic resume but is not serialized or checked.
3. Are artifact checks strong without rejecting legitimate evidence? Check the
   duplicate-attempt and same-stream draw-range rules, parent/child identity,
   shared-invalidity replay veto, and the distinction between a verified parent
   and an unqualified child. Identify any caller-controlled field that can still
   create replay authority.
4. Does the pure controller still satisfy the all-survivor objective? Verify
   that multiple epsilon values at one `L` remain distinct, directional repair
   children preserve exact `L` and mass, repair priority cannot steal a reserved
   unrelated candidate, and a later `L` cannot replace an inconclusive result.
5. Is the reference guide now consistent with the plan and implementation?
   Look specifically for any remaining language that makes the legacy selector,
   a selected candidate, or a compatibility route sound like a second active
   tuner. Do not demand that deferred P2/P3/P5 migration be described as done.
6. Is the R4 plan honest about what remains? The current public numerical
   ordinary and fixed-transport implementations still own independent
   selection/handoff code, and no TensorFlow/TFP/XLA qualification or posterior
   campaign was run. A pure-controller or artifact test must not be interpreted
   as convergence, posterior validity, sampler superiority, default readiness,
   or numerical adapter parity.
7. Are the focused tests discriminating enough to fail bad implementations for
   budget double counting, stale hashes, cross-scope replay, duplicate evidence,
   persisted resume reordering, and unqualified repair handoff? Recommend only
   tests that answer a stated contract gap.

## Required response

Return findings first, ordered by severity. For each material issue classify it
as `confirmed defect`, `unsupported assumption`, `design choice`, `missing
evidence`, or `no finding`; cite the exact path and symbol/line, explain the
failure mode, and give the smallest repair. Distinguish a deferred numerical
qualification from a defect in the completed controller boundary. State which
R4 changes are correct, what remains open for P2/P3/P5, and whether any true
continuation veto exists. End with exactly one line:

`VERDICT: AGREE` or `VERDICT: REVISE`
