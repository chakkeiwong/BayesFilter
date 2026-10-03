# HMC candidate-set unification R3 P0/P1 execution note

Date: 2026-09-12
Plan: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`
Plan revision: R3
Inspected commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`

## Evidence contract

The engineering question is whether one single-scope controller can retain a
broad measured `(L, epsilon)` set, preserve immutable evidence lineage, and
execute a candidate-specific same-`L` repair before later exploration. The
baseline is the current registry and ordinary queue, which still exposes
separate authoritative ordinary and fixed-transport routes. The promotion
criterion for this phase is deterministic state-machine evidence: exact IDs and
hashes, closed-cohort ordering, fresh child verification, reserve accounting,
typed incomplete status, and scope-bound replay. Acceptance and short numerical
diagnostics are explanatory only here. This phase makes no convergence,
posterior, superiority, or GPU/XLA claim. The preserved artifact is the source
tree plus the focused test reports under this run root.

## Skeptical pre-run audit

The plan was amended from R2 to R3 before implementation. The audit found that
`final_status` and `completion_status` were described in overlapping ways for
ordinary budget exhaustion, blocked directional repair, infrastructure pause,
and shared invalidity. R3 gives those fields explicit precedence and states that
verified members do not imply a complete campaign. It also makes parent-reserve
release and child fresh allocation an executable oracle. No long sampler or GPU
run was started.

## P0 inventory

`scripts/inventory_hmc_tuning_routes.py --check` passed with 18 discovered
routes, no stale registry entries, and no unclassified entries. The active
registry still contains authoritative `tune_hmc_kernel` and
`tune_fixed_transport_hmc_kernel`; this is migration debt to address in P3.
The ordinary migration audit completed and wrote its existing inventory under
`docs/plans/artifacts/ordinary-hmc-migration-debt-2026-09-03/`. The downstream
consumer audit was read-only; no MacroFinance or dsge_hmc files were changed.

The dependency inventory confirms that candidate discovery, fixed-transport
selection, generic orchestration, operational grids, dispatch, replay, and
the legacy `select_fixed_transport_candidate_set` helper are separate surfaces.
The new controller is therefore kept independent of the TensorFlow adapters
until its lifecycle is tested.

## P1 result

Added `bayesfilter/inference/hmc_candidate_set_tuning.py` with immutable scope,
candidate, receipt, repair, and work-item records; deterministic cohorts;
deferred repair insertion; same-`L` epsilon children; explicit fresh
verification attempts; reserve release/allocation events; and R3 status
precedence. Added `hmc_candidate_set_artifacts.py` for atomic JSON output,
ordinary SHA-256 result checks, and scope/member replay authority.

Focused controller tests: `13 passed` in
`tests/test_hmc_candidate_set_tuning.py` and
`tests/test_hmc_candidate_set_artifacts.py`.

The tests include the MacroFinance failure shape, inconclusive evidence,
same-`L` directional repair priority, immutable hashes, parent/child lineage,
fresh receipts, reserve accounting, unfunded repair status, interruption and
resume, shared invalidity, infrastructure pause, artifact tampering, and
cross-scope replay rejection.

## Remaining work

P2 must bind the controller to typed TensorFlow/TFP preparation and numerical
adapters without changing public defaults. P3 must migrate dispatch/exports and
retire the active helper authority. P4 must rewrite and render the guidebook.
P5 must qualify each adapter and run the terminal dependency/replay/document
checks. No candidate-set controller result from this phase is numerical or
scientific evidence.

## P4 documentation check

The generated route table was regenerated and its check passed. The full
guidebook compiled to
`book/main.pdf` (556 pages, 2.4 MB) under this run root. The PDF text contains
the new candidate-controller, same-`L` repair, and inconclusive-evidence
language in chapters 21 and 21b. The build retains three pre-existing undefined
bibliography citations (`Afshar2015`, `Gorinova2020`, and `Pakman2014`); those
are recorded as a baseline documentation limitation rather than attributed to
this migration.

The combined focused regression command passed 67 tests with two TensorFlow
Probability deprecation warnings. Public numerical adapter qualification and
active-route replacement remain intentionally incomplete.

The deterministic MacroFinance-shaped smoke is preserved in
`controller-smoke-trace.json` and `controller-smoke-result.json`. It records
root `(L=3, epsilon=0.25)` directional evidence, a child `(L=3,
epsilon=0.50)`, and a later exploratory `L=9`; the child measurement and fresh
verification occur before the later-L measurement. The root receipt, child
hash, reserve events, and qualified repair status are all present in the
checksummed result.
