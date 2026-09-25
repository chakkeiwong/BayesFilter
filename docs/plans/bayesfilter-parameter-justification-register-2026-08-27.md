# Parameter Justification Register (2026-08-27)

Owner finding #3: serious per-scope tuning drift. The LEDH Per-Scope Tuning
Rule in `CLAUDE.md` already required this register; it was never built, so
inherited defaults propagated into a board labeled "production".

**The rule, restated:** every claim-bearing LEDH run requires an offline
tuning artifact for the exact model/target, route/reset family,
horizon/prepared-data regime, particle count, dimensions, dtype/backend,
chunk policy, and route-specific control family used by that run. Any
changed bound field is a NEW tuning scope. A setting selected for another
model, route, or horizon is a warm-start candidate ONLY and must never be
treated as a universal or inherited default.

**Status legend**
- `TUNED(scope)` — calibration artifact exists for exactly this scope
- `WARM(scope)` — value came from another scope; candidate only, no claim
- `INHERITED` — code default, no provenance, no evaluation
- `ASSERTED` — hardcoded literal at a call site, not even a named default
- `DERIVED` — value follows from a derivation on file (not a free parameter)
- `OWNER` — owner decision on file (family policy, not per-scope tuning)

## A. Complete parameter inventory (every free parameter in the call chain)

### A.1 Flow / lifecycle

| # | Parameter | Q3 board value | Status | Notes |
|---|---|---|---|---|
| P01 | `flow_substeps` | 16 | INHERITED | runner literal; signature default is 24 — the board silently disagrees with the code default |
| P02 | `particle_count` | 1008 | WARM(historical) | 1008 from the historical N; never re-justified per scope |
| P03 | UKF `alpha` | 1.0 | INHERITED | sigma-point spread |
| P04 | UKF `beta` | 2.0 | INHERITED | sigma-point kurtosis weight |
| P05 | UKF `kappa` | 0.0 | INHERITED | secondary scaling |
| P06 | UKF `jitter` | DEFAULT_JITTER | INHERITED | Cholesky jitter in the lifecycle |

### A.2 Sinkhorn / OT reset — THE UNTUNED CORE (owner finding #1)

| # | Parameter | Q3 board value | Status | Notes |
|---|---|---|---|---|
| P07 | `epsilon` | 2.0 | INHERITED | **measured bias driver**: -0.650 @ 2.0, -0.257 @ 0.5, -0.734 @ 0.1 (dlgssm T=50). Untuned on ALL SIX rows |
| P08 | `sinkhorn_steps` | 8 | INHERITED | no convergence check; small-eps needs more |
| P09 | `balance_steps` | 8 | INHERITED | no convergence check |
| P10 | `ridge` (reset) | 1e-5 absolute | INHERITED | Q2 Curve 4 measured this as NOMINAL-ONLY on the TF32 lane (4-5 orders below roundoff scale); relative replacement derived, not landed |
| P11 | reset dtype cast | float32 | INHERITED | filter call site hard-casts; couples every "f64" cell to f32 reset islands and drives small-eps underflow |
| P12 | `transport_plan_mode` | dense | INHERITED | chunk policy unexercised at board N |
| P13 | marginal tolerance | 1e-4 | INHERITED | no veto wired for unconverged Sinkhorn |

### A.3 Dual-cap / trust region

| # | Parameter | Q3 board value | Status | Notes |
|---|---|---|---|---|
| P14 | `dual_cap_enabled` | True | OWNER | required mechanism (finding #2) |
| P15 | `trust_region_enabled` | True | OWNER | required mechanism (finding #2) |
| P16 | `trust_region_lm_damping` | 1e-2 | TUNED(austria, on-claim) | Q2 Curve 3 non-harm; **B2 violation**: tuned on claim data |
| P17 | `trust_region_radius` | 0.5 | WARM(austria) | Q2 Curve 2 promotion criterion FAILED; explicitly a warm start |
| P18 | `trust_region_lm_scale_floor` | 1e-4 | ASSERTED | call-site literal; signature default elsewhere is 1e-6 — two different values in one program |
| P19 | `dual_cap_diagonal_steps` | 4 | OWNER | 08-07 family spec |
| P20 | `dual_cap_diagonal_strength` | 0.2 | OWNER | 08-07 family spec |
| P21 | `dual_cap_pairwise_steps` | 4 | OWNER | 08-07 family spec |
| P22 | `dual_cap_pairwise_strength` | 0.02 | OWNER | 08-07 family spec |
| P23 | `pairwise_particle_rms_cap` | 2.0 | OWNER | 08-07 family spec |
| P24 | `coordinate_cap` | 0.98 | OWNER | 08-07 family spec |
| P25 | `coordinate_cap_power` | 8 | ASSERTED | score-lane call-site literal |
| P26 | `floor` (diagonal) | 1e-5 | ASSERTED | score-lane literal; module default 1e-6 |
| P27 | `pairwise_floor` | 1e-5 | ASSERTED | score-lane literal; module default 1e-6 |
| P28 | `RELATIVE_PSD_FLOOR` | 1e-12 | DERIVED | 2026-08-27 response-curve calibration; non-harm verified |

### A.4 Annealed telescope

| # | Parameter | Q3 board value | Status | Notes |
|---|---|---|---|---|
| P29 | `temper_stages` / `annealed_stages` | 4 (austria), 1 (others) | TUNED(austria, on-claim) | Q2 Curve 1; **B2 violation** |
| P30 | `flow_prior_cap` | 8.0 (austria), inf (others) | TUNED(austria, on-claim) | Q2 Curve 1; measured lane-defining |
| P31 | `annealed_seed` | 17 | INHERITED | fixed-index resampling convention |
| P32 | `resample_seed` | = value seed | INHERITED | coupling to the value seed is unexamined |

### A.5 Evaluation-side (not algorithm, but claim-bearing)

| # | Parameter | Q3 board value | Status | Notes |
|---|---|---|---|---|
| P33 | value seeds | 8 | INHERITED | too few for uncertainty analysis (gap D2) |
| P34 | score seeds | 8 (4 on KSC) | INHERITED | KSC reduction undeclared in the plan |
| P35 | exact-score FD step `h` | 1e-6 | INHERITED | the dlgssm "exact" score is an FD reference, not analytic |

## B. Drift summary

- 35 free parameters in the claim-bearing call chain.
- **TUNED for the scope actually claimed: ZERO.** The three `TUNED` rows are
  Austria-only AND were tuned on the claim observations (B2 violation), so
  even Austria has no clean artifact.
- 5 scope-specific tunings owed (dlgssm, linear2d, predator-prey, KSC,
  gen-SV) x the full control family. Austria owes a re-run on disjoint data.
- **6 ASSERTED literals** (P18, P25, P26, P27 and the two floors) where a
  call-site number silently overrides a different module default. P18 is the
  worst: 1e-4 at the call site vs 1e-6 in the signature, inside the same
  production program.
- **1 self-contradiction**: P01 `flow_substeps` = 16 in the runner vs 24 in
  the signature default. Neither is justified; they disagree.

## C. Enforcement (so this cannot drift again)

The rule existed and was not enforced. Three mechanisms, mirroring the
configuration-status-first remedy:

1. **This register is the single source of parameter provenance.** Every
   parameter of the production program appears here with a status. A
   parameter absent from this register may not appear in a claim-bearing
   run.
2. **Machine-checked completeness**: a governance test asserts that every
   free parameter of `canonical_value_and_diagnostics`,
   `canonical_value_and_analytical_score`, `_restore_cloud_primal`, and
   `higher_moment_shape_jvp` has a row here, and that no row is missing a
   status. A new parameter added to any signature fails the gate until it is
   registered.
3. **Claim gate**: a cell may be claim-bearing only if every parameter it
   uses is `TUNED(its own scope)`, `DERIVED`, or `OWNER`. Any `INHERITED`,
   `ASSERTED`, or `WARM` parameter forces the cell to report
   `UNTUNED — not claim-bearing`, which the report builder already renders
   before any number.

Under gate 3, **every cell of the current Q3 board is non-claim-bearing**.
The board remains valid as mechanism evidence (the program runs, is
oracle-exact, and is finite everywhere); it carries no per-model performance
claim. That is the honest status and matches what the numbers show — the
dlgssm value error of 0.870 nats is dominated by untuned `epsilon`, not by
the algorithm's design.

## D. Required work before any claim

R2-TUNE (per scope, 6 scopes): enumerate the control family (P01, P02, P07,
P08, P09, P10, P11, P29, P30 at minimum), tune on data disjoint from the
claim partition, with promotion criteria (value-bias budget against the
exact reference where one exists; ESS floor; Sinkhorn marginal-convergence
veto; claim-scale Fisher pass for score cells), and emit one artifact per
scope that the board's `tuning` field cites by path.

R2-LITERAL (cheap, do first): resolve the 6 ASSERTED literals and the P01
contradiction — either promote to registry constants with a derivation, or
expose them as tuned parameters. No campaign needed; this is a code-hygiene
pass that removes silent overrides.
