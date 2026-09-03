# BayesFilter Consolidated Issue Registry

Date: 2026-08-21

Purpose: single authoritative list of every defect, gap, and open decision
surfaced in the 2026-08-18..21 investigation, for the owner's process
review. Each row states verification status. Classifications follow the
plain-language policy: wrong-relative-to-claim / unsupported / not-checked.

## A. Algorithm-fidelity gaps (spec vs executing code)

| # | Issue | Evidence | Status |
|---|---|---|---|
| A1 | NeuTra batch lane (`cubature_genut_batch_tf.py` + Austria adapter) contains NO particle flow: bootstrap proposal (RK4+noise), likelihood weighting, then OT reset. It is a bootstrap PF with post-hoc redistribution, not LEDH-PFPF-OT. | Adapter read 2026-08-21; zero flow references | Verified. All 2026-08 root-cause campaigns validated this lane. |
| A2 | No UKF / per-particle covariance lifecycle exists in ANY lane. Li(17) Alg. 1's EKF/UKF predict->update recursion (documented as an implementation contract in `ch19c_dpf_implementation_literature.tex:230-340`, incl. resample-the-triple rule) has no corresponding code. No sigma-point call anywhere in the DPF flow path. | Repo-wide grep 2026-08-21 | Verified. The "UKF initialization faithfully implemented" reassurance is wrong relative to that claim. |
| A3 | Austria flow-lane callbacks feed placeholder Gaussians (`ledh_pfpf_genut_model_callbacks_tf.py:448-461`): `initial_covariance = eye(18)`, `transition_covariance = eye(18)`. Aggravation (other agent, confirmed): at `ledh_pfpf_genut_initial_rqmc_tf.py:738-741` the identity is also the SAMPLING covariance of the pre-flow noise (`pre_flow = prior_mean + noise @ chol(transition_covariance)^T`), so a fix changes the realized proposal distribution, not just flow algebra. | Both agents, independently | Verified. |
| A4 | `transition_matrix = eye(18)` where faithful LEDH wiring wants the RK4 transition Jacobian (or sigma-point equivalent). Transition MEAN is faithful (float64 RK4 per ancestor); observation side is modeled (linear extraction, `100*exp(2*theta_2)*I_9`). Defect confined to the three state-side Gaussian inputs. | Other agent, consistent with source read | Verified. |
| A5 | Dual-cap trust-region "general implementation" was a lane fork: general single-cloud `higher_moment_shape_jvp` exists and is real, but the claim-bearing batch lane held a diagonal-only reimplementation lacking pairwise moments and both caps. Value-side port + parity oracle landed 2026-08-20; the batch JVP (score side) remains diagonal-only. | Call-chain audit 2026-08-20 | Partially repaired. |
| A6 | Trust-region/LM stabilization existed since 2026-08-15 but was frozen at zero in the wired route; rejected once against a question it could never pass (identity repair), never re-asked the stability question. | Provenance archaeology 2026-08-20 | Root cause of NaN exposure; controls now evaluated (C1). |

## B. Statistical / filtering problems

| # | Issue | Evidence | Status |
|---|---|---|---|
| B1 | Weight degeneracy in the NeuTra lane: ESS collapses to ~23/1008 at informative steps (bootstrap proposal, by construction of A1). | CPU probe 2026-08-20 (descriptive lane) | Verified mechanism; exact in-route per-step ESS not yet instrumented (Class A task). |
| B2 | Higher-moment targets estimated from the degenerate weighted cloud BETWEEN weighting and reset: garbage 4th-moment targets (kurtosis ~37 vs Gaussian 3) drive the correction. | P0/P2 diagnostics + probe | Verified. |
| B3 | Undamped Gauss-Newton moment correction demands 44-56 sigma displacements; universal method fragility (reproduced on clean Gaussian toy), not SIR-specific; no healthy regime at frozen Austria scope. | Toy demo + R2 arms | Verified. Cap tames it; whether the corrected computation is statistically meaningful is NOT checked. |
| B4 | Flow lane ESS also "small relative to N" per its own campaign notes; sibling branch measured 0.14% proposal-target overlap (row-coverage collapse, ESS 11/8192). Consistent with A2-A4: flow runs at partial strength on placeholder covariances. | 2026-08-17 result note; commit 8b7296c9 | Verified descriptively; flow-lane Austria per-step ESS profile unread from artifacts (cheap next step). |
| B5 | No accuracy reference exists: capped/uncapped/eager/graph/XLA/CPU values span ~5 nats and no experiment can currently say which is closest to the true likelihood. | Cross-artifact comparison | Open; needs a designed reference experiment. |
| B6 | Whether the higher-moment correction earns its place AT ALL at this scope (steps=0 was valid/finite everywhere) has never been tested on accuracy. | Campaign record | Open; cheapest potentially decisive experiment. |

## C. Numerical / compiler defects

| # | Issue | Evidence | Status |
|---|---|---|---|
| C1 | XLA+TF32 NaN at T=20,steps=4: TF32-seeded arithmetic degradation blowing up in the unprotected Stage D solves; guard fired correctly. Trust controls (warm-start 0.5/1e-2) remove it and restore exact within-mode identity under XLA. | Localization + R2 campaigns | Mechanism verified; controls UNJUSTIFIED pending calibration (C4). |
| C2 | Non-XLA graph mode: grappler rewrites value-only and JVP-carrying graphs differently -> within-mode value/score program split (0.562 at T=20; reproduces at T=3,steps=0). `disable_meta_optimizer=True` restores bitwise identity. | Graph localization campaign | Verified; owner scope decision pending (E-decisions). |
| C3 | Cross-mode value drift: eager-GPU TF32 value ~2.3 nats from CPU; XLA-TF32-off lands near CPU. TF32-dominated; interpretation not checked. | Artifacts | Recorded, uninterpreted (needs B5 framework). |
| C4 | Class C hyperparameters unjustified across the board: trust radius (0.5 AND the former 0.0), LM damping, ridge 1e-5 (absolute, inherited), dual-cap constants (0.98/8, 2.0). Calibration protocol defined (R6), not executed. | R6 plan section | Open. |
| C5 | Three formerly-unridged/unchecked Choleskys in the batch route: check-only guards + min-diagonal diagnostic landed 2026-08-20 (9 tests pass). Ridge (numerics-altering) deferred to calibration. | R3 | Guarded; ridging open. |

## D. Latent hazards (will bite if unaddressed)

| # | Issue | Status |
|---|---|---|
| D1 | Enabling the newly ported pairwise/coordinate controls on the value route while the batch JVP lacks them would split value/score programs by construction. Currently protected only by default-off + documentation. Runtime guard (~10 lines in `batch_finite_value_score`) proposed, NOT yet implemented. | Open — small, should be next code change. |
| D2 | Any A3/A4 repair changes the realized proposal distribution (per the sampling-covariance finding), invalidating flow-lane tuning scopes and historical comparability; must be planned as a new-scope campaign, not a patch. | Planning constraint. |
| D3 | `_symmetric_sylvester_ops.so` ABI break (environment): target-factory tests fail on Austria construction for unrelated reasons. | Known, quarantined. |

## E. Process / governance failures (the owner's review subject)

| # | Failure | Instances |
|---|---|---|
| E1 | Audits verified EXISTENCE (of functions, documents, capabilities) instead of CALL CHAINS from claim-bearing endpoints. | 4: trust controls (existed, unwired), dual-cap (existed, forked), flow (existed, absent from NeuTra lane), UKF (documented, absent everywhere). |
| E2 | Doc-code "consistency" audits checked doc-vs-paper fidelity and code-internal consistency, never bound each documented algorithm ARROW to executing code. ch19c contains the exact warnings describing the defects that shipped. | The UKF lifecycle arrow had no code on the other end; no audit executed the arrow. |
| E3 | Lane limitations never carried as first-class caveats: "batch_diagonal_candidate" honestly disclaimed dual-cap but NEVER stated "no flow, bootstrap proposal"; the flow lane never stated "identity covariances, no UKF". Omission, not fabrication — every individual document is locally honest. | Months of campaigns attached conclusions to lanes whose relation to the target algorithm was unstated. |
| E4 | Reassurances relayed as verified ("general implementation, audit-verified"; "UKF faithfully implemented") were wrong relative to their claims; narrative verification was accepted where executable evidence (parity/wiring tests) was required. | Codex report; UKF reassurance. |
| E5 | Rejected candidates not recorded with the question they failed (trust region rejected as identity-repair, never re-asked as stability mechanism). | Policy now amended (reversed-burden, C-class rules). |
| E6 | Opportunity cost: multi-month debugging program (value/score identity, compiler modes, NaN) executed against the interim lane. The findings are real and transferable (compiler and numerics issues affect any lane), but the priority ordering would have been different had A1/A2 been known. | The deepest cost of E1-E3. |

## Standing remedies already in place

- Reversed-burden guardrail policy + Class C justification rule + call-chain
  audit rule: canonical in claudecodex, installed to home and repo targets.
- Class A/B: diagnostics serialization, Cholesky guards, parity oracle,
  fork-regrowth signature test — landed with passing gates.
- R2 evidence: trust controls remove the NaN (warm-start, uncalibrated).
- Proposed next artifact: algorithm-contract conformance matrix — rows =
  ch19c's numbered Alg. 1 steps, columns = lanes, cells = file:line
  verified-by-execution or ABSENT. Would have caught E1-E3 mechanically.

## Priority-ordered open queue (for the owner's re-planning)

1. D1 runtime guard (trivial, closes a live trap).
2. Read flow-lane Austria per-step ESS from existing 2026-08-06 artifacts
   (zero compute; decides how sick the flow lane is).
3. Conformance matrix (one page; makes E1-E3 impossible to repeat silently).
4. UKF-prediction discriminating experiment: sigma-point covariance into the
   flow (machinery exists in `bayesfilter/nonlinear/sigma_points_tf.py`),
   measure ESS vs identity-covariance baseline on frozen Austria scope.
   Note D2: new tuning scope by construction.
5. B6 steps=0 accuracy experiment + B5 reference design.
6. R6 calibration campaign; then R7 (JVP port, wiring switch, full ladder).
7. Owner decisions: graph-mode scope; TF32 posture; whether the NeuTra lane
   should exist at all vs differentiating the flow lane.
