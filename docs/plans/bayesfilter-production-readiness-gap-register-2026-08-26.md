# Production-Readiness Gap Register (2026-08-26)

Question: what stands between the current state and PRODUCTION-LEVEL
value and score cells for every model, i.e. a leaderboard whose every
cell is (a) the production program, (b) covered by a per-scope tuning
artifact, (c) estimand-anchored, (d) statistically honest. Every gap
below carries its evidence anchor; nothing here is speculative.

Definition of production-level (per cell):
1. program = the declared production algorithm for that scope
   (LEDH-PFPF core + smooth OT/Contract-E reset + the owner-directed
   dual-cap family where its calibration says so), no silent variants;
2. a per-scope tuning artifact covering the full control family, tuned
   on data DISJOINT from the claim partition (LEDH tuning rule);
3. estimand anchoring: exact reference where linear; a budgeted
   reference arm or identity gate (Fisher) at claim scale elsewhere;
4. replication that is real (independent seeds — fixed 2026-08-25) and,
   for any ranking language, a predeclared uncertainty analysis.

## A. Program gaps (cross-cutting)

| # | Gap | Evidence | Remedy |
|---|---|---|---|
| A1 | Dual-cap trust region OFF in the value filter default and never exercised in board cells; dual-cap S7 exists in the score lane but likewise unexercised | filter signature default; Q3 ledger 2026-08-25 | Owner decision + per-scope tuning of the dual-cap family (constants have owner rationale, Curve 5; activation per scope is a tuning output, not a global toggle) |
| A2 | Entropic-transport value bias: N-PERSISTENT (-0.65/-0.69/-0.78 at N=252/1008/4032, dlgssm T=50) and epsilon-sensitive (-0.65 @ eps=2.0 -> -0.26 @ eps=0.5; degraded at eps=0.1/24 iters) | 2026-08-25 diagnostics | Per-scope eps tuning WITH a Sinkhorn marginal-convergence veto (unconverged small-eps cells must be rejected, not averaged); debiased/eps-schedule transport is a candidate under its own contract |
| A3 | Production score bias grows with T (-0.12/-0.55/-1.39 at T=10/25/50, dlgssm) — the derivative of A2's biased value program | 2026-08-25 diagnostics | Expected to co-shrink with A2 tuning; verify against exact score on anchor rows; if Q5 needs more, bias-corrected score estimators are a new contract |
| A4 | Reset always casts to float32 at the filter call site — small-eps kernels underflow in f32; also couples every "f64" value cell to f32 reset islands | filter line ~348; eps=0.1 degradation | Revisit per scope during eps tuning; dtype-of-reset becomes a recorded tuning field |
| A5 | Reset ridge 1e-5 absolute = nominal-only on the TF32 lane; relative replacement derived but not landed | Q2 Curve 4 | Land the relative ridge under its Class-C non-harm contract (healthy/pathological fixture pair exists) or record owner acceptance of the measured risk |
| A6 | Only the Austria model factory is dtype-parameterized; the repo's GPU f32/TF32 production execution target cannot run the other five models | dtype param added 2026-08-24 for Austria only | Parameterize the five factories (mechanical, same pattern) + per-model f32 battery arms |
| A7 | Annealed-mode score keeps FIXED realized resampling indices in the tangent (Austria score cell) — a bias term by construction, distinct from A3 | Q1.2 convention; board note | Per-scope choice between annealed and reset-only score programs, decided by a claim-scale Fisher gate, in the tuning campaign |

## B. Tuning gaps (per-scope rule)

| # | Gap | Evidence |
|---|---|---|
| B1 | NO per-scope tuning artifact for linear2d, dlgssm, predator-prey, KSC, gen-SV — every control (eps, Sinkhorn/balance iters, flow_substeps, k/c, dual-cap family, ridge) is an inherited default | board configuration-status table (17x UNTUNED) |
| B2 | Austria's only calibrations (k/c, damping) were tuned ON the frozen CLAIM observations (ESS screen on claim data) — the tuning rule requires disjoint tuning partitions for claim-bearing use | Q2 Curve-1 runner uses the frozen target |
| B3 | The control family per scope is not even enumerated as a tuning contract for any model but Austria | Q2 plan covers Austria curves only |

Remedy for B: one tuning campaign per model scope, on FRESH tuning data
(simulated from the model law, or a held-out partition disjoint from the
frozen claim tensors), each with: enumerated control family, promotion
criteria (value-bias budget on anchored rows; ESS floor; claim-scale
Fisher pass for score; Sinkhorn-convergence veto), and a tuning artifact
the leaderboard cells cite in their `tuning` field. Austria re-confirms
k/c on disjoint data (cheap: the surface was sharp).

## C. Estimand / reference gaps

| # | Gap | Evidence |
|---|---|---|
| C1 | linear2d row has NO score cell (the only row where the exact score reference is free) | board |
| C2 | Nonlinear rows (pp, ksc, gsv, austria) have no value reference arm — cross-arm tables are descriptive with no anchor; pp shows an unexplained 9.6-nat canonical-vs-UKF-GF gap with a collapsed bootstrap (spread 39) that anchors nothing | board value cells |
| C3 | Fisher identity gates exist for all five stochastic models but at short horizons/small scopes; only Austria has a claim-scale Fisher artifact (and it used annealing WITHOUT the reset — not the current production score program) | test file + Q2 Curve 7 |
| C4 | gen-SV has no frozen dataset lineage (board row simulates ad hoc at seed 501) | target factory: not bridged |
| C5 | Austria production score spread is 310 over 8 seeds — with that variance the cell is uninformative for Q5 uses; variance source not yet decomposed (direction scale? annealing? N?) | board |
| C6 | Initial-covariance provenance: board rows use eye(dim) labeled model_exact; correct for Austria/lgssm frozen targets (scale 1.0 recorded), NOT verified for pp/ksc/gsv | callbacks in Q3 runner |

Remedy for C: add the linear score cell (free); freeze a gen-SV dataset
with SHA lineage; per nonlinear row add ONE budgeted reference arm
(large-N annealed bootstrap or SQMC with MCSE, run once per row) so
value cells get an anchor; claim-scale Fisher gates per model under the
production score program; verify/record initial-covariance provenance
per row.

## D. Statistical-layer gaps

| # | Gap | Evidence |
|---|---|---|
| D1 | Arms are not seed-paired (canonical uses hashed generator streams; bootstrap uses np streams) — paired uncertainty analysis is impossible on the current cells | Q3 runner |
| D2 | No predeclared uncertainty analysis exists, so the board can never legitimately rank; that is currently declared honestly but is a gap if ranking is ever wanted | inference-status table |
| D3 | KSC production score at T=1000 costs ~40 min for 4 seeds — multi-seed claim-scale score cells need either a streaming reset in the score lane or an explicit reduced-seed policy with MCSE reporting | prod board wall times |

## E. Comparator gaps (slice B)

| # | Gap |
|---|---|
| E1 | SGQF comparator exists for SIR and gen-SV but is not hooked into the board (data-convention bridge unwritten) |
| E2 | zhao-cui reference arm not hooked |

## Remediation phases (dependency order)

- R1 Program completion (CPU/cheap): A5 ridge contract, A6 dtype
  parameterization + f32 batteries, C1 linear score cell, C4 gen-SV
  freeze, C6 provenance check, D1 seed pairing in the runner.
  ~1 session, no GPU campaigns.
- R2 Per-scope tuning campaigns (GPU, the core): B1-B3 + A1/A2/A4/A7
  resolved per scope as tuning outputs; disjoint tuning data; one
  tuning artifact per model. Budget: ~1-2 GPU-days total (Austria's
  surface sharpness suggests coarse ladders suffice).
- R3 References + identity gates (GPU, parallel to R2 partly): C2
  reference arms, C3 claim-scale Fisher under the production score
  program, C5 Austria score-variance decomposition. ~1 GPU-day.
- R4 Board regeneration + statistical layer: paired 16-seed rerun of
  all cells under tuned configs (D1), predeclared uncertainty analysis
  if ranking is wanted (D2), slice-B comparators (E1-E2), report with
  every `tuning` field pointing at an artifact. ~0.5 GPU-day.

Only after R4 does the leaderboard meet the production-level definition
above. Nothing in R1-R4 requires an owner decision EXCEPT: A1 (dual-cap
default/activation policy), A5 if the ridge change is declined, and any
ranking ambition in D2 (which sets R4's analysis scope).
