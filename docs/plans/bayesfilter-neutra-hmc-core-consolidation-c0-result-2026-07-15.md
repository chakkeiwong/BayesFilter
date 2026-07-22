# NeuTra HMC Core Consolidation Phase C0 Result

Date: 2026-07-15  
Decision: `PASS_C0_ROUTE_AND_POLICY_CONTRACT`

## Outcome

The canonical policy identifier is `bayesfilter_neutra_sequential_hmc_v1`.
NeuTra HMC routes are now governed by a versioned route ledger with persistent
source discovery, exact-one classification, active core-binding checks, and
explicit historical/reference exceptions.

The current active claim-bearing surface is the 2026-07-15 LGSSM gap-closure
campaign and its CLI. The 2026-07-13 serious campaign, its 2026-07-14 delegated
target-specific HMC phases, and the legacy fixed-transport tuning API are
historical/superseded for new serious NeuTra claims. Phase 18-20 mechanics and
reference helpers remain explicit non-claim exceptions.

## Artifacts And Checks

- Ledger:
  `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/route_ledger.json`.
- Enforcement: `bayesfilter/inference/neutra_hmc_policy.py`.
- Negative tests cover unledgered routes, stale/duplicate paths, missing core
  symbols, missing policy binding, and fixed-budget active entry points.
- Focused result: `4 passed`; compile and `git diff --check` pass.
- Plan review: Claude returned two `REVISE` verdicts, the plan was visibly
  patched, and the third review returned `VERDICT: AGREE`.

## Handoff

Proceed to
`docs/plans/bayesfilter-neutra-hmc-core-consolidation-c1-subplan-2026-07-15.md`.
No sampler behavior or historical artifact was changed in C0.
