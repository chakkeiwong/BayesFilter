# Zhao-Cui Predator-Prey Fixed-Variant Active Plan

Date: 2026-07-23
Status: `IMPLEMENTATION_GATES_PASS_NOT_ADMITTED`
Source audit: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-source-audit-2026-07-23.md`
Handoff: `docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-implementation-handoff-2026-07-23.md`

## Research Intent And Evidence Contract

Question: can a predator-prey-specific, parameter-independent Zhao-Cui-derived
proposal branch produce the sealed source-order T20 finite value and the exact
manual score of that same finite program without retained-grid storage?

Candidate: squared-TT/TTSIRT proposal operations compiled offline, with frozen
particles, genealogy, auxiliary categorical laws, and exact proposal-density
corrections; online evaluation uses a graph-native source-order APF recursion.

Exact comparator: none. Same-target fixed SGQF and GenUT values/scores are
descriptive plausibility checks only. The old retained-grid Zhao-Cui result is a
historical negative control, not a baseline or oracle.

Primary pass criterion: deterministic finite value, six finite physical score
coordinates, increment identities, and same-program central-FD agreement for
the sealed program.

Promotion vetoes: wrong target/hash/event order; initial `y0` assimilation;
invalid proposal density or auxiliary law; nonpositive defensive mass;
parameter-dependent prepared branch; retained-grid marker; runtime autodiff or
finite differences; failed score audit; non-finite result; missing GPU/XLA or
memory-growth evidence for a claim run.

Continuation vetoes: the paper/code audit invalidates the correction; the
generated proposal density cannot be evaluated; the sealed dataset is corrupt;
the route cannot avoid retained-grid storage; the extension classification is
not admissible under leaderboard policy; or the bounded campaign budget is
exhausted.

Repair triggers: valid-harness ESS, fit, conditioning, or score failure. Repair
or retune on fresh calibration data; never tune on the sealed claim data.

Explanatory diagnostics: ESS by time, log-weight spread, fit/conditional errors,
rank/degree/L1 behavior, compile/runtime, allocator current/peak, and descriptive
SGQF/GenUT gaps.

Nonclaims: exact likelihood, unbiased pseudo-marginal estimator, source-faithful
assembled route, posterior or HMC validity, high-dimensional model validation,
default readiness, or statistical superiority.

Artifact root for future serious runs:
`docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/`.
Every attempt must use a fresh subdirectory.

## Skeptical Plan Audit

| Risk | Resolution |
| --- | --- |
| Wrong baseline | SGQF and GenUT are descriptive; retained-grid is a negative control. |
| Proxy promoted | FD, fit loss, ESS, and one-seed gaps are diagnostics/vetoes, not superiority evidence. |
| Hidden target drift | Bind target ID, seed, hashes, T20, physical parameter order, and `x0 -> transition -> y1` into repository-issued identity. |
| Wrong scalar | Implement a new `[T+1,N,2]` / `[T,2]` source-order program; do not modify the initial-observation-first program. |
| Partial score | Differentiate normalized previous weights recursively and compare the complete scalar with FD in tests only. |
| Stale defaults | Author/rung-2 controls are warm starts; predator-prey rank, degree, map, defensive mass, ridge, L1, fit budget, and auxiliary law require scope-specific tuning. |
| Environment mismatch | CPU is mechanics/reference only. Claim-bearing execution is FP32, TF32, GPU, XLA, with verified memory growth before initialization. |
| Non-answering artifact | A serious result must include value/score, increments, identities, hashes, diagnostics, frozen controls, device/memory/XLA data, command, Git state, seeds, and wall time. |
| Misleading pass | A score-perfect but collapsed proposal is not promotable; a high-ESS but score-wrong program is invalid. |

Audit verdict: `PASS_FOR_PHASES_1_2`. Proposal-quality tuning and leaderboard
admission remain gated. The current tree has no competing predator-prey fixed
route. Unrelated rung-2 and four-filter worktree files are out of scope.

Final evidence is recorded in
`docs/plans/bayesfilter-zhao-cui-predator-prey-fixed-variant-result-2026-07-23.md`.
The route remains unadmitted because the assembled program is classified as
`extension_or_invention`, not a source-faithful adaptive Zhao-Cui route.

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- | --- |
| Sealed dataset/target | Existing source-order dataset factory and handoff | reviewed target | Hash or event-order mismatch | Exact hash/shape test |
| Frozen branch | HMC differentiability requirement | `fixed_hmc_adaptation` | Omitted theta dependence | Identity and theta-invariance tests |
| Source-order APF scalar | Project derivation in handoff | hypothesis under test | Normalization/ancestor correction error | Independent tiny scalar |
| Manual score | Model density-score methods plus normalized-weight recursion | hypothesis under test | Local score mistaken for total score | Six-coordinate FD test |
| Squared-TT/KR proposal | Zhao-Cui Eq. 13, Proposition 2, Algorithm 3 and pinned code | source-backed operations | Density/measure mismatch | Conditional normalization and generation-density parity |
| Rank/degree/map/L1/ridge/tau | Not yet selected for predator-prey | warm-start hypotheses only | Collapse or overfit | Disjoint calibration/validation ladder |
| `N=1002` first GPU arm | Existing same-target GenUT feasibility scope | feasibility convenience | Too noisy for claim | ESS/spread and larger-N predeclared arm |
| FP32/TF32/GPU/XLA | Repository default | required claim backend | Numerical/compile failure | Short GPU smoke after CPU parity |

## Phases And Bounded Budget

1. Implement source-order branch/program and independent scalar tests.
2. Implement analytical recursive score and six-coordinate FD diagnostics.
3. Add source-order TTSIRT branch compilation and proposal-mechanics tests.
4. Add target-specific offline fit/tuning protocol with positive L1 arms and
   disjoint calibration/validation/audit designs.
5. Run CPU FP64 reference and a short GPU/XLA smoke with memory growth.
6. Only after tuning and untouched N>1000 evidence, consider route registry and
   leaderboard admission.

This turn's compute budget is at most 12 focused test/smoke attempts, no single
attempt over five minutes, and at most one short GPU/XLA smoke. A long tuning
ladder, N=5000/10000 claim run, or leaderboard promotion requires a completed
tuning artifact and remains a later serious run under this same scientific
contract.

## Pre-Mortem

The code could pass a scalar test while assimilating the wrong observation;
the independent fixture must make initial-observation-first observably
different. It could pass FD while using autodiff internally; source inspection
and provenance tests forbid that. It could compile while proposals collapse;
ESS/spread are promotion veto diagnostics. It could show good ESS by evaluating
the wrong proposal density; inverse/forward and generated-density parity are
required before tuning. It could fit the sealed data; tuning inputs must be
disjoint and frozen before any untouched claim evaluation.
