# NeuTra Banana HMC Repair Plan (2026-08-16)

## Research Intent Ledger

| Field | Predeclared statement |
|---|---|
| Main question | Is the banana HMC failure caused by the learned transport geometry, the warm-up initial-state bank, or the shared fixed-kernel/controller mechanics? |
| Frozen learned candidate | The target-specific root-preserving `(32,32)` transport retrained with seed `15`, 6,000 updates, peak `LR=5e-4`, and the fixed 3,000-update schedule horizon. Its proposal audit must pass before any HMC arm runs. |
| Arm A: learned/original bank | Frozen learned transport with the original iid-normal four-chain bank and the original HMC tuning grid. This is a reproduction arm for the r3 HMC failure. |
| Arm B: learned/central bank | The same frozen learned transport and HMC policy, but with a deterministic near-central bank: rows `0`, `+0.25 e_0`, `-0.25 e_0`, and `+0.25 e_1` in z coordinates. This isolates warm-up initialization without changing transport or target. |
| Arm C: exact analytic transport | A source-independent analytic banana map `theta_0=z_0`, `theta_1=z_1+c(z_0^2-1)`, `theta_j=z_j` for `j>=2`, with unit Jacobian and exact pullback score. It is a mechanics/geometry positive control, not a learned-transport candidate. |
| Primary diagnosis | Arm A reproduces the r3 health failure; Arm B passes while A fails; or Arm C fails. These outcomes distinguish start sensitivity, learned-geometry sensitivity, and controller/kernel failure respectively. A single arm cannot establish a universal explanation. |
| HMC gates | Each arm has its own target-scope tuning artifact, no `L=1`, shared sequential HMC controller, warm-up R-hat `<=1.05`, retained R-hat `<=1.01`, ESS `>=400`, finite-state/target/score/log-accept, movement, energy, and retained exact-law screens. |
| Hard vetoes | Failed proposal audit for learned transport, nonfinite HMC values, positive native divergence when exposed, no chain movement, warm-up/retained cap without readiness, retained exact-law failure, invalid adapter binding, or invalid artifact hashes. |
| Nonclaims | No universal HMC kernel, no statistical superiority, no production/default readiness, no SSL-LSTM transfer, and no claim that a positive control proves the learned transport correct. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Seed-15 learned transport | r3 used seed 15 for the conditional HMC arm and its 6,000-update proposal audit passed | Reproduces the exact candidate whose HMC failed | A new retrain could differ if stateless replay is broken | State hash and 131,072-draw proposal audit | Reviewed warm start |
| Original iid-normal bank | r3 terminal manifest and prior campaign helper | Reproduction comparator | A warm-up basin may depend on starts | Arm A health and R-hat | Reviewed comparator |
| Central bank | New diagnostic hypothesis | Tests whether warm-up initialization causes the numerical failure | Central starts may be unrepresentative of production use | Arm B proposal/HMC gates and bank manifest | Hypothesis under test |
| Exact analytic map | Derived from the declared banana transform and unit Jacobian | Provides a known correct geometry/controller control | It does not test learned transport capacity | Arm C exact-law and HMC gates | Reviewed positive control |
| Tuning grid `L=(3,5,10,15,20,25)` | Existing target-specific policy; `L=1` forbidden | Reproduces prior kernel-selection scope | Grid may omit a usable trajectory length | Fresh per-arm tuning verification | Reviewed policy |
| Four chains and sequential budgets | Shared NeuTra HMC policy | Required for rank-normalized multi-chain checks | Bounded evidence only | R-hat/ESS/cap diagnostics | Policy default |

## Evidence Contract

| Item | Predeclared value |
|---|---|
| Baseline/comparator | Arm A, learned transport plus original bank and fixed identity-z mass |
| Promotion criterion | None. This is a diagnosis campaign; it may nominate a repair but does not promote a new default. |
| Diagnostic interpretation | Arm A reproduces/fails; Arm B changes only starts; Arm C changes only the transport map. The result is classified as start-sensitive, learned-geometry-sensitive, or unresolved/controller-sensitive only when the corresponding hard screens support it. |
| Veto diagnostics | Nonfinite log-acceptance is a hard numerical veto even when acceptance, states, or ESS look favorable. |
| Explanatory diagnostics | Acceptance, loss, proposal ESS, runtime, selected `L`, step size, and standardized discrepancies. |
| Non-conclusions | A passing analytic map does not certify the learned map; a passing central bank does not prove the original bank invalid; one campaign does not establish HMC correctness for SSL-LSTM. |
| Artifact | Terminal discovery root `docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3/` and matched-kernel cross-over root `docs/plans/artifacts/neutra-banana-hmc-matched-kernel-2026-08-16-r1/`, each with manifests, results, and hashes. |

## Skeptical Plan Audit

| Risk | Disposition |
|---|---|
| Training failure is mistaken for HMC failure | Vetoed: the learned transport must pass the unchanged 131,072-draw exact-law proposal audit before Arm A/B HMC. |
| Start-bank change is confounded with tuning | Vetoed: Arm A and B each create their own target-scope tuning artifact using the same grid and policy; only the bank differs. |
| Analytic map is treated as learned evidence | Vetoed: Arm C is labeled a mechanics/geometry positive control and cannot promote NeuTra. |
| Acceptance or ESS masks NaN log-acceptance | Vetoed: `log_accept_ratio_all_finite` is a hard health requirement. |
| Reusing r3 artifacts silently changes provenance | Vetoed: seed-15 training is replayed and hashed in the fresh r1 root, using the exact r3 audit partition `59015`; r3 remains historical evidence. |
| Relaxing retained gates rescues a failing arm | Vetoed: controller R-hat, ESS, finite, movement, energy, and exact-law gates are unchanged. |
| HMC diagnosis is overinterpreted from one arm | Vetoed: outcome language is conditional on the predeclared three-arm pattern. |

Audit verdict: the plan isolates the three live explanations with one frozen
learned candidate, one start-bank change, and one exact analytic geometry
control. It preserves the existing HMC gates and is fit for a bounded GPU run.

## Pre-Rerun Review Of `r2`

The first implementation reached Arm A and produced a valid diagnostic, then
stopped before Arm B because the new analytic control omitted the repository
`parameter_dim` adapter contract. The partial `r2` root is retained as
debugging evidence only. Arm A showed warm-up/retained health and convergence
passing but retained exact-law failure; no Arm B/C conclusion is drawn. The
analytic adapter contract is corrected before the terminal rerun.

## Post-Discovery Causal Audit And Cross-Over

The terminal three-arm `r3` run found learned/original failing, learned/central
passing, and analytic/original passing. However, fresh per-arm tuning selected
`L=5`, step `0.8361329642` for learned/original and `L=10`, step
`0.7709722546` for learned/central. The plan's claim that Arm B isolates only
the bank is therefore unsupported by the diagonal comparison alone.

The required repair is a matched-kernel 2x2 cross-over with no retuning:

1. preserve the existing original-bank/`L=5` and central-bank/`L=10` cells;
2. run original bank with the frozen central-selected `L=10` kernel; and
3. run central bank with the frozen original-selected `L=5` kernel.

The same learned transport, target, controller, warm-up/retained seeds, and all
validity gates remain fixed. If bank determines passage under both kernels,
start sensitivity is supported for this transport. If kernel determines
passage under both banks, kernel sensitivity is supported. If only the
diagonal cells differ, the interaction remains unresolved. This cross-over is
diagnostic and does not promote either kernel or bank.

## L=10 Confirmation

The cross-over supports a target-specific `L=10` candidate, but its retained
screen used only 2,000 draws per chain. The terminal confirmation therefore
freezes the central-selected kernel (`L=10`, step `0.7709722545680272`) and
runs both the original and central banks for `5,000` retained draws per chain,
with the same 2,000-draw warm-up and unchanged hard gates. No retuning,
training, mass adaptation, or threshold change is allowed. A pass on both
banks is confirmation for this frozen learned transport only; any failure is
preserved as a candidate veto, not repaired by relaxing gates.

## Execution

1. Retrain seed 15 for 6,000 updates with the fixed r3 schedule and verify its proposal audit.
2. Build Arm A, B, and C adapters and target-scope manifests.
3. Tune each arm on the fixed grid with per-arm output directories.
4. Run sequential HMC only after each arm's tuning verification passes.
5. Write a result note with the arm decision table, inference-status table, red-team interpretation, and a reset memo.
6. After the terminal result audit, execute the no-retuning 2x2 cross-over and update the result classification.
7. Execute the fixed-`L=10` 5,000-draw-per-chain confirmation on both banks and record its result separately.
