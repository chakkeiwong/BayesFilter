# NeuTra HMC Core Consolidation Phase C2 Subplan

Date: 2026-07-15  
Status: `EXECUTED_AND_CLOSED`

## Objective And Entry

Make the active LGSSM claim-bearing route a compatibility consumer of the C1
controller without changing target, kernel, seed derivation, diagnostics, or
historical tensor archive schema. C1 tests and XLA smoke passed.

## Artifacts And Checks

- import and delegate shared configs and controllers;
- preserve LGSSM target-status detail and flat cumulative archive keys through
  callbacks;
- remove local HMC controller ownership;
- reject reachable local `HamiltonianMonteCarlo` or `sample_chain` bypasses;
- keep historical routes and artifacts unchanged;
- run campaign, core, ledger, negative enforcement, compile, and diff checks.

## Evidence And Handoff

Pass requires behavioral compatibility tests and the discovery ledger to pass.
No historical result is reinterpreted. On pass, engineering consolidation is
complete and S1 may run one fresh target-specific training seed. Stop only for
target/kernel/artifact drift or a failing enforcement invariant.

Suitability review: `PASS`. The wrapper/core boundary preserves the existing
campaign schema and makes bypass mechanically detectable.
