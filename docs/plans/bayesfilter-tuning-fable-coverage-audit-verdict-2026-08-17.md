# Fable Coverage Audit Verdict: Tuning-Test and Posterior-Coverage Adequacy

To: Codex
From: Fable
Date: 2026-08-17
Request: `docs/plans/bayesfilter-tuning-fable-thorough-test-coverage-audit-request-2026-08-17.md`
Plan audited: `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`
(revised 2026-08-17, 654 lines, including the mass-matrix contract and
posterior-oracle adequacy gate)

Boundaries honored: read-only; no source edits; no GPU/CUDA commands; all
executed checks CPU-hidden (`CUDA_VISIBLE_DEVICES=-1`) under conda `tfgpu`
(Python 3.13.13, TF 2.20.0, TFP 0.25.0); dsge_hmc run with
`BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter`; `tests/archive` untouched.
BayesFilter commit `3030d86d` with the known dirty worktree preserved.
Headline: this audit **ran** the plan's focused commands instead of collecting
them, and the run-level results change the picture (Finding F1).

## A. Findings

### F1 — BLOCKING (cross-repo migration gates): the plan's consumer baselines are collection-only, and run-level execution reveals 13 pre-existing contract-drift failures inside plan-named focused gate files

The plan's recorded baselines (plan lines 466-471) say "101 tests collected,
no collection error" (MacroFinance six focused files) and "26 tests collected,
no collection error" (dsge_hmc three contract files). Both replicate. But
running those same files today gives:

- MacroFinance six focused files: **90 passed, 11 failed** (227 s);
- dsge_hmc three contract files: **24 passed, 2 failed** (18 s).

All 13 failures are pre-existing cross-repo contract drift between committed
BayesFilter and committed consumer expectations — none are caused by the dirty
worktree, and none are refactor effects. Three distinct families:

1. **Acceptance-evidence schema v5 drift (8 failures).**
   `bayesfilter/inference/hmc_verification.py:910` (committed) hard-rejects
   any payload whose schema is not `bayesfilter.hmc_acceptance_evidence.v5`;
   the CCMA confirmation fixtures in
   `MacroFinance/tests/test_run_ccma_broad_fixed_metric_l_epsilon_search.py`
   reconstruct payloads that fail this guard (`ValueError: acceptance
   evidence schema mismatch`). One of the plan's six named Phase-5 focused
   files therefore fails at run level today.
2. **Staged-timeout policy payload drift (3 failures).**
   `MacroFinance/tests/test_daily_asset_midas_l10d_bayesfilter_bootstrap_geometry_repair.py:251-252,550`
   binds a caller policy with `policy_id="ccma_phase4y_stage_budget_v1"`; the
   assertion `config.staged_timeout_policy is policy` passes, but the
   BayesFilter `payload()` projection reports the committed default
   `bayesfilter_hmc_emergency_stage_caps_v2`
   (`bayesfilter/inference/hmc_kernel_tuning.py:767` at HEAD). The payload
   does not reflect the bound caller policy — either intentional public-payload
   sanitization or a serialization defect; not adjudicated anywhere.
3. **Fixed-transport grid-policy drift (2 failures).**
   `dsge_hmc/tests/contracts/test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py:187,226`
   expects `policy_selection_rule ==
   "shortest_leapfrog_acceptance_in_band_then_diagnostics"` and
   `full_grid_candidate_count == 49`; committed BayesFilter
   (`bayesfilter/inference/fixed_transport_hmc_grid_policy.py:243`) now
   returns
   `"eligible_trajectory_acceptance_in_band_then_rhat_convergence_then_ess"`
   and 63 candidates. The Phase-4/6 gate file fails today.

Why blocking: the Phase 5 gate ("focused MacroFinance tuning tests ... pass",
plan line 411) and the Phase 4/6 gates (lines 333-337, 423-428) are
unsatisfiable at baseline for reasons the plan does not record. This is the
same defect class as R1's B1/B2 (which the plan fixed for *collection*
baselines) recurring one evidence level up. It blocks the cross-repo migration
gates, not Phases 0-2 implementation. It also demands adjudication before
migration: for each family, decide whether the consumer contract is stale
(update the consumer test) or BayesFilter broke a compatibility promise
(restore/shim), and record which.

### F2 — MATERIAL: the plan's mass-contract command omits the two most relevant existing mass test files

The "mass-specific contract suite" command (plan lines 496-503) runs
`test_hmc_kernel_tuning_geometry.py`, `..._bootstrap.py`,
`..._windowed_mass.py`, and `test_hmc_budget_ladder.py -k 'mass or covariance
or metric or gaussian'`. It does not include
`tests/test_hmc_mass_matrix.py` (14 exact-construction/regularization/
fail-closed/signature tests — precisely mass-contract items 1 and 3) or
`tests/test_hmc_windowed_mass_adaptation.py` (Welford-vs-NumPy reference,
shrinkage-target staleness, covariance floor/condition cap — precisely item
2). Executed as written, the command passes 42 / skips 1 / deselects 73 while
missing the strongest existing mass evidence. Repair: add both files to the
command.

### F3 — MATERIAL: the mass-versus-target holdout lacks the deliberately-bad-mass arm

Plan lines 278-288 compare identity, exact-supplied, and warmup-adapted
metrics. The request (Q3 check 5) also requires "a deliberately bad mass under
one target." This arm is absent. It matters because it is the negative control
for the holdout gate itself: a bad-but-numerically-valid mass (e.g., inverse
covariance used as covariance) still targets the correct density, so holdout
validity should pass while efficiency/repair diagnostics visibly fire — which
is exactly what distinguishes "target-preserving validity criterion" from "the
gate cannot discriminate at all." The fail-closed family (item 3) covers only
invalid inputs, not valid-but-wrong ones. Repair: add the bad-mass arm with
the expectation stated (validity holds or fails visibly; explanatory/repair
diagnostics must fire; no promotion semantics).

### F4 — MATERIAL: the mass contract does not address the NumPy-backed mass construction authority

`bayesfilter/inference/mass_matrix.py` — covariance/precision inversion,
regularization, whitening, eigenvalue summaries — is NumPy throughout
(`import numpy as np`, line 14) and sits on the runtime tuning path via
`hmc.py`'s `PrecomputedMassArtifact`. The repository backend rule names
tuning, candidate selection/admission, and artifact construction as
non-exempt from the TensorFlow/TFP requirement, and the plan's Phase 3
requires "no NumPy numerical path in admitted execution" (line 313) — but the
mass-matrix contract section (lines 222-293) never states whether NumPy mass
construction at tuning time is migration debt to be converted during the
Phase-2 `hmc_geometry.py` extraction, or an accepted host-side boundary. The
extraction will move this exact code; the contract must say which. Repair: one
paragraph in the mass contract declaring the status and (if debt) the
conversion/recording requirement.

### F5 — NONBLOCKING: no nonlinear or hierarchical stress fixture, currently confined by an explicit nonclaim

The only curved/banana targets in the repository live in the NeuTra domain
(e.g., `bayesfilter/inference/neutra_varying_hessian_target.py`,
`tests/test_neutra_banana_predictive_equivalence.py`), outside this
consolidation. The plan's oracle nonclaim (lines 383-385) explicitly excludes
nonlinear targets, so the omission is currently honest. Decision requested by
Q2: the Gaussian-plus-domain fixture set **is sufficient for this refactor**
(structural extraction, parity, and known-target agreement), and is **not
sufficient for any future default promotion of tuning policy numerics** (L
grids, acceptance bands, qualification budgets). The plan's hidden-defaults
row promises target-specific tests before promotion but names no fixture.
Recommended addition (nonblocking now): a sentence in Phase 3 that promoting
robust-route numeric defaults requires at least one non-Gaussian stress
fixture (the repository's varying-Hessian/banana target is the natural
candidate) or an owner waiver.

### F6 — NONBLOCKING: identity-only transports in the current fixed-transport tests

Every transport in `tests/test_fixed_transport_hmc_tuning.py` is an identity
(`counting_identity_transport.v1`, `mass_policy == "fixed_identity_z"`), and
no affine transport appears anywhere in the BayesFilter fixed-transport test
family. The plan already states "an identity transport is a control, not
sufficient coverage" (line 372) and makes the affine holdout oracle item 3.
Confirmed as correctly-recorded open work; noted here because the dsge_hmc
side does have its own affine-posterior contract
(`dsge_hmc/tests/contracts/test_fixed_affine_posterior.py`), which partially
compensates at the consumer level but does not test the BayesFilter tuner.

### F7 — CONFIRMED (strengths verified by execution, not just reading)

- The trajectory family has a real negative control where acceptance alone
  would select the wrong L:
  `tests/test_hmc_kernel_tuning_frozen_step_trajectory.py:602`
  (`high_acceptance_underreach_cannot_pass`) plus its Phase-23 nomination
  variants — mechanics-level (scripted runner), exactly the request's Q3
  check 6 shape.
- Mass construction fail-closed behavior is genuinely tested:
  `tests/test_hmc_mass_matrix.py` (nonfinite, indefinite-regularization
  metadata, signature/dimension/schema/corruption rejections) and
  `tests/test_hmc_windowed_mass_adaptation.py` (Welford vs `np.cov`
  reference at line 113, covariance floor/condition cap at line 299,
  stale-shrinkage rejection).
- Real-TFP execution exists at every stage boundary: bootstrap (line 825),
  windowed mass (lines 1459-1603), budget ladder (line 1909), fixed-transport
  XLA route (line 418) — smoke-scale, correctly not treated as validity
  evidence by the plan.
- The plan's evidence taxonomy is clean: collection vs mechanics vs
  target-validity are separated, the posterior-oracle test is treated as an
  open gap until it exists (lines 518-520), and the missing MacroFinance
  robust-driver test is recorded as a gap, not a pass (lines 549-553).
- `tests/test_hmc_tuning_posterior_oracle.py` does not exist — consistent
  with the plan's own reporting.
- No active consumer family is missing from Phases 5-6 (verified against my
  R1 symbol-level sweep; budget-ladder and generic-orchestration families are
  now present at lines 400-405).

## B. Posterior-Fixture Matrix

| Problem | Current fixture(s) | Reference/oracle | Role | Planned closure | Gap |
|---|---|---|---|---|---|
| Isotropic baseline | `_ToyGaussianAdapter` variants: `tests/test_hmc_kernel_tuning_fixed_mass_step.py:50-54` (log p = -0.5·Σθ²), `tests/test_hmc_budget_ladder.py:29-49`, tiny-Gaussian real-TFP runs in windowed-mass/bootstrap tests | Correct-by-construction score; **no analytic moment check anywhere** | mechanics + real-TFP smoke | Oracle gate keeps isotropic as control | Acceptable: superseded by oracle case |
| Shifted correlated Gaussian | none (no current test uses nonzero mean + non-diagonal Σ as pass/fail) | closed-form μ, Σ | target-oracle | `tests/test_hmc_tuning_posterior_oracle.py` items 1-2 (lines 347-367) | Open deliverable, correctly recorded |
| Rotated/anisotropic Gaussian | one non-diagonal negative Hessian `[[4,.25],[.25,3]]` in `tests/test_hmc_windowed_mass_adaptation.py:221`; budget ladder uses `np.eye(2)`/`np.diag` only | independent covariance calc | mechanics | Mass windowed-estimator test item 2 requires a rotated non-diagonal fixture (lines 264-266) | Open deliverable, correctly recorded |
| Ill-conditioned Gaussian | `tests/test_hmc_mass_matrix.py:34-76` (indefinite regularization + metadata, nonfinite fail-closed); `tests/test_hmc_windowed_mass_adaptation.py:299` (floor/condition cap) | constructed matrices | mechanics (fail-closed) | Mass item 3 extends | Covered at construction level |
| Affine transformed Gaussian | none in BayesFilter (all transports identity, F6); consumer-side `dsge_hmc/tests/contracts/test_fixed_affine_posterior.py` | transformed closed form | target-oracle | Oracle item 3 (lines 368-372) | Open deliverable, correctly recorded |
| Wrong-target controls | scope/schema/signature rejections only (`tests/test_frozen_kernel_validation.py`, mass artifact staleness) — identity-level, not numeric wrongness | deliberate wrong μ/Σ/Jacobian/score | target-oracle (negative control) | Oracle item 4 (lines 373-375) | Open deliverable, correctly recorded |
| Nonlinear curved target | NeuTra domain only (banana/varying-Hessian, outside consolidation) | none for canonical tuners | stress diagnostic | none planned; explicit nonclaim (lines 383-385) | F5: required before any default promotion, not for this refactor |
| Hierarchical/funnel | none | none | stress diagnostic | none planned; covered by the same nonclaim | Same as F5; omission is explicit |
| Domain-specific synthetic | MacroFinance: matched-DGP contract (`bayesfilter_macrofinance_matched_dgp_contract.py`), CCMA exact-target test, cross-country synthetic-recovery/randomized-truth tests; dsge_hmc: `test_bgs_hmc_synthetic_recovery_bayesfilter_runner.py`, `test_bgs_mass_geometry_tf.py`, `test_fixed_affine_posterior.py`, BGS fresh-target-validity | model-specific synthetic truths | consumer integration | Phases 5-6 migrate these | Covered on consumer side; run-level status subject to F1 drift |

Decision (Q2): Gaussian + domain fixtures are sufficient for the refactor's
claims as scoped by the plan's nonclaims; a nonlinear stress fixture becomes
required only when default-promotion language about tuning numerics is
introduced (F5).

## C. Joint Mass/Epsilon/L Matrix

The plan describes a **staged conditional** search, never claiming jointness:
`geometry -> bootstrap -> mass -> epsilon/L -> verification` (line 114), with
mass updates invalidating epsilon/trajectory context and forcing fresh seeds
and typed handoff (lines 248-250). Q3 check 1 is satisfied. Two epsilon
policies coexist: the monolith route tunes epsilon at fixed mass then selects
L at frozen step with repair handoffs; the robust route re-runs dual averaging
per L from the windowed epsilon
(`bayesfilter/inference/hmc_robust_broad_grid.py:259-292,521-527`) with
domain-separated seeds per stage (`_seed(root, "dual_averaging"|"repair"|
"qualification", L, index)`, lines 63-72). The plan requires the request to
bind the "epsilon/trajectory candidate policy" (line 48) but does not name the
two policies; nonblocking, worth one sentence in Phase 3.

| Control | Candidate policy | Stage | Retuned after mass change? | Calibration data | Holdout data | Primary criterion | Vetoes | Repair trigger | Artifact identity | Current test | Planned test | Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Initial mass/geometry | hint precedence: neg-Hessian > covariance > scales | geometry | n/a (origin) | MAP/hints | none | finite PD kernel | nonfinite, shape, indefinite-without-policy | fallback policy recorded | mass signature + hint provenance | `test_hmc_kernel_tuning_geometry.py` (12 tests, precedence + fail-closed), `test_hmc_mass_matrix.py` | mass items 1,3 | closed (construction) |
| Adapted mass | windowed Welford + shrinkage, floor/condition cap; `fixed_identity` alternative | windowed stage | n/a (is the change) | warmup draws | none yet | stage pass + telemetry | stale shrinkage target, invalid retained diagnostics, signature mismatch | retry with private pair seeds | adapted-mass signature | `test_hmc_kernel_tuning_windowed_mass.py` (30+ incl. real-TFP), `test_hmc_windowed_mass_adaptation.py` (Welford ref, floor/cap) | mass item 2 (rotated fixture) | rotated fixture open |
| Mass **selection** (identity vs exact vs adapted) | holdout comparison | Phase 2/3 gate | — | matched seeds/starts | untouched holdout | target-preserving holdout validity | divergence, nonfinite, R-hat/ESS gates | failed mass = repair, not HMC evidence | all three arms bound | **none** (no moment check exists anywhere) | mass item 5 (lines 278-288) | open; **F3: no deliberately-bad-mass arm** |
| Epsilon | monolith: DA at fixed mass; robust: DA per L from windowed epsilon | fixed-mass step / per-L | **yes** — both routes retune after windowed mass; robust seeds per L | tuning seeds | repair screens use fresh seeds | acceptance band vs target | nonfinite step trace, evidence invalidity, cost stops | band-scaled epsilon repair (`_repair_one_l`, factor 1.25, ≤5) | seed + config in payload | `test_hmc_kernel_tuning_fixed_mass_step.py` (fake-runner mechanics), `test_hmc_fixed_mass_step_tuning.py` | oracle holdout exercises the chain end-to-end | robust campaign stages themselves untested (selector-only tests) |
| L / trajectory | monolith: frozen-step trajectory window + tau; robust: fixed grid (3,5,9,13,18,25), min-bulk-ESS selection | trajectory stage / qualification | uses declared epsilon per policy above | trajectory candidates | robust: 500-draw qualification (screen, not holdout) | trajectory-window relation + acceptance; robust: suitability then max min-bulk-ESS | below-window, divergence>0, R-hat>1.05, nonfinite | underreach → repair handoff (floors/caps step) | candidate signature per L | `test_hmc_kernel_tuning_frozen_step_trajectory.py` incl. **negative control at :602** (high acceptance cannot pass underreach); robust selector tests | oracle holdout; Phase 3 generalizes frozen grid | real-TFP-level L-interaction test open (oracle closes) |
| Untouched verification | fresh seeds, frozen artifact | final | consumes final mass/epsilon/L | none (must be untouched) | independent chains | analytic moment agreement + validity gates | any validity failure | back to repair, no data reuse | artifact hash recorded | **none** | oracle items 2-3 | open deliverable, correctly recorded |

Interaction summary: interactions are tested at mechanics level (trajectory
window × acceptance; mass signature × epsilon stage handoff via
`test_windowed_mass_stage_validates_adapter_and_mass_signatures`), and the
full mass→epsilon→L→holdout interaction is tested nowhere today — the oracle
gate plus mass item 5 are the planned closure, and the plan correctly reports
them open. Mass construction is covered while mass *selection* is not
(flagged exactly as Q3 requested); epsilon/acceptance interaction with L has
a mechanics negative control but no real-TFP counterpart yet.

## D. MacroFinance Coverage Map

Callers verified against tests (file counts from the R1 `--no-ignore` sweep;
run/collection status measured this session):

| Caller family | Interface today | Tests found | Status |
|---|---|---|---|
| MIDAS ordinary/map/bootstrap-geometry/fixed-mass (`daily_asset_midas_*`) | `tune_hmc_kernel` + geometry/bootstrap stage APIs + `HMCStagedTimeoutPolicy` | `test_daily_asset_midas_bayesfilter_owned_tuning_execution.py` (source call-count + mock), `..._bounded_tuning_repaired_stack.py`, `..._l10c...` (source scan), `..._l10d...` (20 tests), plus map/eom/phase5/7/8 tuning-execution tests (~15 more files) | l10d: **3 run-level failures** (F1 family 2); others in focused set pass |
| MIDAS robust broad grid | `tune_hmc_kernel_robust_broad_grid` (1 driver) | **none** (`tests/*robust_broad*` empty) | Correctly recorded Phase-5 gap (plan lines 549-553) |
| CCMA fixed-metric + confirmation | `run_fixed_metric_grid_search` + `evaluate_hmc_acceptance_evidence` | `test_run_ccma_broad_fixed_metric_l_epsilon_search.py` (34 tests) + trajectory-discriminator + statistical-epsilon-repair + operational tests | **8 run-level failures** (F1 family 1: v5 schema guard) |
| CCMA operational grid | `run_operational_broad_grid` (1 script) | `test_ccma_operational_broad_l_epsilon_neighbor_guard.py`, `test_run_ccma_hmc_operational_recovery.py`, `test_ccma_operational_exact_target.py` | Not in focused set; run status not measured |
| Cross-country generic orchestration | `orchestrate_generic_hmc_tuning` | `test_cross_country_multi_asset_bayesfilter_owned_hmc_client.py`, `..._mass_preconditioner.py` + ~18 `test_cross_country_*hmc*` (synthetic recovery, randomized truth, XLA ladders) | In Phase 5 item 6; run status not measured |
| Mixed-frequency budget ladder | `run_fixed_mass_hmc_tuning_budget_ladder` | `test_mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase5t_real_tuning_loop.py`, `..._svd_finite_reject.py`, `..._phase5v_heldout_validation.py` | In Phase 5 item 5; run status not measured |
| One-country ZLB/analytic | `tune_hmc_kernel` + map-mass | `test_one_country_analytic_map_mass_matrix.py`, `..._analytic_hmc_adapter.py`, `..._hmc_analytic_gradient_hessian.py`, `test_one_country_zlb_ns_estimation.py` | ZLB estimation test is in the 38 full-suite collection errors (pandas) |
| Two-currency fixed-metric/NeuTra | fixed-metric + NeuTra routes | `test_two_currency_double_zlb_dz5_neutra_fixed_metric_grid.py` (passes in focused run) + 4 more dz5 tuning tests | Focused member passes |
| Private/source imports | `hmc_kernel_tuning` source scans, `_mass_artifact_signature` (30-35 files) | source-level assertions only (e.g. `test_daily_asset_midas_bayesfilter_owned_tuning_execution.py:167-171`) | Behavior-level replacement tests are open Phase-5 work, correctly recorded |

Full-suite baseline: 4,252 collected / 38 errors replicated in R1. Causes
split: missing `pandas` in `tfgpu` (MIDAS/portfolio/two-currency families,
~30+) and order-dependent "BayesFilter checkout is unavailable" RuntimeErrors
(`daily_asset_midas_full_model_phase2_bayesfilter_architecture.py:33-37`
resolves at import time; single-file collection succeeds, full-suite order
fails). Not a pass; the plan records both causes and requires root-cause or
owner waiver (line 470) — correct. Root-causing the order dependence remains
open (smallest artifact: pairwise `--collect-only` bisection).

## E. dsge_hmc Coverage Map

| Caller family | Interface today | Tests found | Status |
|---|---|---|---|
| BGS Stage-C grid | `run_fixed_metric_grid_search` + lineage types | `tests/contracts/test_bgs_bayesfilter_stage_c_grid_tuning.py` (8 tests: callback/aggregate) | Passes in focused run |
| Rotemberg fixed-transport/NeuTra | `tune_fixed_transport_hmc_kernel`, grid-policy spec | `test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py` (11), `test_rotemberg_fixed_neutra_xla_gate.py` | **2 run-level failures** (F1 family 3: selection-rule + 63-vs-49 candidate count) |
| Explicit-state fixed-mass scripts (11 + round380 adapter) | private `_mass_artifact_signature`, `_BootstrapFixedMassLatentValueScoreAdapter` | 17 `test_rotemberg_public_explicit_state_*` contract tests exist, but **no dsge test imports the private BayesFilter names** — scripts fail only when run | Private-name replacement coverage is open Phase-6 work, correctly recorded |
| Path resolution | `ensure_bayesfilter_on_path` / `BAYESFILTER_ROOT` | `tests/contracts/test_bayesfilter_path_resolution.py` | Present; env var required and now in all plan commands |
| BGS mass/geometry + synthetic recovery | geometry/mass stage APIs | `test_bgs_mass_geometry_tf.py`, `test_bgs_hmc_synthetic_recovery_bayesfilter_runner.py`, `test_bgs_hmc_fresh_target_validity.py` | Present (not run this session) |
| Rotemberg affine-posterior/component | dsge-side contracts | `test_fixed_affine_posterior.py`, `test_bgs_pydsge_phase04_affine_parity.py`, 4rr bounded-tuning/budget-ladder contracts | Present; partially compensates F6 at consumer level |

Command hygiene verified: focused command with `BAYESFILTER_ROOT` collects 26
and runs 24/2 (F1); the configured full suite requires `--ignore=tests/archive`
(module-level scipy segfault in `tests/archive/test_rotemberg_nk.py:151`,
reproduced in R1) and the plan records the omission and forbids calling
archive tests passing (lines 567-573). The 457-file contracts directory means
focused contracts genuinely cannot certify all production callers — the
plan's full-suite gate plus the F1 adjudication is the right closure.

## F. Required Plan Repairs

| Finding | Repair | Blocks |
|---|---|---|
| F1 | (a) Extend the Phase-0 baseline from collection-only to **run-level** for the plan-named focused files, recording: MacroFinance six files 90/11, dsge three files 24/2 (2026-08-17, `tfgpu`, CPU-hidden). (b) Add a drift-adjudication deliverable to Phase 0/1: for each family — acceptance-evidence v5 guard (`hmc_verification.py:910`) vs CCMA fixtures; staged-timeout `payload()` policy-id projection (`hmc_kernel_tuning.py:767`) vs l10d expectation; grid-policy selection rule/count (`fixed_transport_hmc_grid_policy.py:243`) vs Rotemberg relaunch contract — decide "consumer contract stale → update consumer test" or "BayesFilter compatibility break → restore/shim", with owner sign-off recorded. (c) Reword Phase 4/5/6 gates to "pass relative to the recorded run-level baseline after adjudicated drift is resolved." | Cross-repo migration gates (Phases 4-6). Does not block Phases 0-2 extraction. |
| F2 | Add `tests/test_hmc_mass_matrix.py` and `tests/test_hmc_windowed_mass_adaptation.py` to the mass-contract command (plan lines 496-503). | Mass-contract gate fidelity (Phase 2). |
| F3 | Add a deliberately-bad-but-valid mass arm to the mass-versus-target holdout (plan lines 278-288), with declared expectations: holdout validity may pass, explanatory/repair diagnostics must fire, and no promotion inference either way. | Target-validity testing (Phase 2/3 holdout design). |
| F4 | State in the mass contract whether `mass_matrix.py`'s NumPy construction path is migration debt to be converted in the Phase-2 `hmc_geometry.py` extraction (backend rule names tuning/artifact construction as non-exempt) or a recorded host-boundary exception with owner approval. | Phase-2 extraction of the mass family. |
| F5 | One sentence in Phase 3: promoting robust-route numeric defaults (L grid, bands, 500-rung) requires at least one non-Gaussian stress fixture or an explicit owner waiver. | Default promotion only. |

F6 needs no plan change (already stated at line 372); it is closed by oracle
item 3 when implemented.

## G. Evidence and Nonclaims

- **Hard veto evidence (this session):** the 13 run-level focused-gate
  failures with exact assertion diffs; the archive segfault (R1); the
  fail-closed GPU tests under CPU hiding (R1).
- **Descriptive/engineering evidence:** fast-check block 1 = 90 passed
  (66 s, includes two dirty-worktree test files), block 2 = 65 passed (15 s);
  mass-filtered command = 42 passed / 1 skipped / 73 deselected (36 s);
  MacroFinance focused = 90/11 (227 s); dsge focused = 24/2 (18 s). All
  CPU-hidden `tfgpu`; GPU intentionally hidden.
- **Collection evidence only:** BayesFilter 7,452/3 (R2), MacroFinance full
  4,252/38 (R1), dsge focused 26 (R1). Collection is not run evidence — F1 is
  the proof.
- **Statistical evidence:** none was generated; nothing here ranks any tuner,
  mass, or kernel.
- **Unimplemented gates (correctly open in the plan):**
  `tests/test_hmc_tuning_posterior_oracle.py`; the mass-versus-target holdout;
  the rotated-Gaussian windowed fixture; affine-transport holdout; negative
  controls; route guard/inventory script; MacroFinance robust-driver test;
  private-import replacement tests.
- **This audit does not establish:** posterior correctness, convergence,
  sampler validity or superiority, adequacy of any current default, GPU
  behavior, or that any consumer's science is affected by the F1 drifts —
  only that the named contract tests fail against committed code today.
  Run-level status was measured only for the plan-named focused files; other
  consumer test families were inventoried but not executed.

Verdict rationale: the request's REVISE triggers are "lacks a required
fixture" (F3 bad-mass arm; F2 command omission) and "treats
collection/mechanics as target-validity evidence" — the plan's prose does
not, but its recorded consumer baselines do exactly that one level down:
collection-only baselines presented where the gates need run-level truth, and
run-level truth currently contains 13 pre-existing failures (F1). No active
consumer is missed, and the staged mass/epsilon/L retuning is properly
defined, so the plan's architecture stands; the gates and two fixture-design
details need the listed repairs.

`AUDIT_VERDICT: REVISE`

`PLAN_VERDICT: REVISE`
