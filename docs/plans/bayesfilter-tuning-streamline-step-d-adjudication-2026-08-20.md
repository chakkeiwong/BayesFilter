# BayesFilter Tuning Streamline: Step D Call-Site Adjudication and Migration Plan

Date: 2026-08-20
Status: ACTIVE — all five Section-5 decisions resolved by owner on 2026-08-20:
(1) add public bootstrap-adapter facade now, preserving class-identity string;
(2) freeze old lineage, migrate producers; fresh runs create new lineage;
(3) dsge_hmc retained_validation and mass_injection_audit both frozen as
documented migration debt; (4=5) DZ5 canary and MIDAS phase10 validation both
grandfathered as historical evidence on current routes.
Governing documents:
- `docs/plans/bayesfilter-tuning-streamline-claude-code-handoff-2026-08-19.md` (Sections 9–11)
- `docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md` (Phases 5–7)
- `docs/plans/bayesfilter-tuning-streamline-consumer-repair-result-2026-08-20.md` (Steps A–C result)

## 1. Inventory Provenance

Section 9's required `rg` inventory was run on 2026-08-20 with explicit path
arguments and artifact-tree exclusions (`results/`, `models/`, `docs/`,
`__pycache__/`, `build/` — non-Python artifact trees only; every `.py`
source/test/script location scanned). Raw outputs preserved at the session
scratchpad (`macrofinance-inventory.txt`: 146 hits / ~50 files;
`dsge-hmc-inventory.txt`: 43 hits / 16 files). Every file was classified by
reading docstrings, artifact-role labels, and producer/consumer links — not by
filename.

Substring caveat: many `_mass_artifact_signature` hits are false positives on
the payload key string `"adapted_mass_artifact_signature"`; those are
documentation-only for the symbol regardless of file status.

## 2. SDK Facts That Drive the Plan

- Public facade exists for the mass signature:
  `bayesfilter.inference.hmc_artifacts.mass_artifact_signature` (also exported
  in `bayesfilter/inference/__init__.py:322`).
- Public `build_fixed_mass_hmc_adapter` exists
  (`hmc_budget_ladder.py:1525`, exported `__init__.py:449`).
- **SDK GAP**: no public facade exists for
  `_build_bootstrap_fixed_mass_adapter` /
  `_BootstrapFixedMassLatentValueScoreAdapter`. ~16 MacroFinance files and one
  dsge_hmc claim-adjacent file need it. The private class name is embedded in
  adapter-signature payloads (`hmc_kernel_tuning.py:13587`, `:18997`) and
  consumers pin exact signatures (`EXPECTED_PHASE4_ADAPTER_SIGNATURE`;
  dsge_hmc `mass_injection_audit.py:231`), so a facade MUST preserve the
  class-identity string. This touches handoff stop condition 5.
- `_mass_artifact_signature` is defined in two module homes
  (`hmc_kernel_tuning.py:13551` and `hmc_budget_ladder.py:2631`); byte-level
  semantic identity must be confirmed once before consolidating consumers.

## 3. dsge_hmc Adjudication (16 files)

No tuning-interface migration is required anywhere in dsge_hmc. The historical
tuning-route executions are all frozen one-shots.

| Class | Files | Action |
| --- | --- | --- |
| (a) claim-bearing | `src/dsge_hmc/experiment_adapters/rotemberg_round380_neutra.py`; scripts `..._mass_artifact_repair.py`, `..._bayesfilter_candidate_campaign.py`, `..._bayesfilter_candidate_refinement.py` | Import swap `_mass_artifact_signature` → public facade. Signature use is identity verification only; no tuning route involved. Frozen r16 artifacts are SHA-pinned — swaps affect future reruns only. |
| (a/b) FLAGGED | `..._retained_validation.py` | See Decision 3. Uses `_build_fixed_mass_hmc_adapter` (public replacement exists) + `_mass_artifact_signature`; retains scientific draws for the frozen r16 candidate yet self-labels `SEED_COMPLETED_DIAGNOSTIC_ONLY`. |
| (b) diagnostic, stays | `run_bgs_bayesfilter_phase08_stage_c_grid_tuning.py` (plan allowance satisfied: aggregate policy preserved, `promotion_status: not_authorized_until_fresh_confirmation`, C2_CLOSED statuses); `run_rotemberg_fixed_neutra_bayesfilter_tuning_smoke.py` (already calls canonical `tune_fixed_transport_hmc_kernel:425`); `..._gap_target_domain_diagnostic.py` | Leave with nonclaims. Optional: update the smoke script's `current_public_replacements` advisory strings (:381-385) to name the two canonical interfaces. |
| (b) FLAGGED | `..._mass_injection_audit.py` | See Decision 4. Its :231 private-class-path string is load-bearing in a recomputed signature binding; import swap would invalidate `latent_adapter_signature_mass_bound`. |
| (c) historical | `..._fixed_mass_tuning.py` (sole executor of the budget-ladder route), `..._fixed_kernel_epsilon_screen.py`, `..._fixed_kernel_epsilon_replication.py`, `..._fixed_kernel_short_screen.py`, `..._fixed_mass_bootstrap.py` | No change. Frozen 2026-08-08 repair-plan one-shots; refuse to rerun; explicit nonclaims. |
| (d) tests | `test_rotemberg_fixed_neutra_xla_gate.py` (enforces canonical binding), `test_bgs_bayesfilter_stage_c_grid_tuning.py` (pins the grid route AND its non-promotion contract) | None unless the pinned runner changes. |

## 4. MacroFinance Adjudication (~50 files)

### Group A — migrate to canonical interface (claim-bearing)

1. `daily_asset_midas_robust_broad_grid_tuning.py` — imports/calls
   `tune_hmc_kernel_robust_broad_grid` (:48, :235). CONFIRMED claim-bearing:
   its private result is loaded by
   `daily_asset_midas_bayesfilter_retained_synthetic_hmc.py:36-37`, which
   rebuilds geometry/adapted mass from it for the retained 2026-08-15 run.
   Migrate to `tune_hmc_kernel`. Blast radius: pinned handoff signatures in
   the retained runner (`EXPECTED_OPERATIONAL_MASS_SIGNATURE`,
   `EXPECTED_FINAL_SIGNATURE`) — see Decision 2.
2. `mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase4_step_trajectory.py`
   — calls `run_fixed_mass_hmc_tuning_budget_ladder` (:820) and
   `orchestrate_generic_hmc_tuning` (:1205); writes
   `tuning_authority: "bayesfilter.inference.run_fixed_mass_hmc_tuning_budget_ladder"`
   (:1427) which authorizes Phase 5T real execution. Migrate to
   `tune_hmc_kernel` **atomically with**:
3. `..._phase5t_real_tuning_loop.py` — authority-string contract only
   (:159, :617, :780; no historical call), plus tests
   `test_..._phase5t_real_tuning_loop.py:129,298`,
   `test_..._phase5v_heldout_validation.py:125`,
   `test_..._svd_finite_reject.py:519`.
4. `cross_country_multi_asset_bayesfilter_owned_hmc_client.py` — calls
   `orchestrate_generic_hmc_tuning` 3x (:998, :1137, :1199); the final call
   produces the frozen Phase-5 launch configuration. Migrate to
   `tune_hmc_kernel`. Keep the :1497/:1534 negative source-boundary
   assertions (tests pin them).
5. `daily_asset_midas_bayesfilter_retained_synthetic_hmc.py` — swap
   `_mass_artifact_signature` to public facade now; the
   `_build_bootstrap_fixed_mass_adapter` swap is BLOCKED on the SDK facade
   (Decision 1).

### Group B — behavior-preserving private-import swaps (diagnostic files)

Fourteen CCMA `phase5*/phase10*` diagnostics + `daily_asset_midas_eom_bootstrap_hmc_diagnostic.py`
access `hkt._mass_artifact_signature` / `hkt._build_bootstrap_fixed_mass_adapter`.
All explicitly diagnostic by docstring. Swap the signature import to the
public facade; bootstrap-builder swap blocked on Decision 1.
`daily_asset_midas_phase10_synthetic_hmc_validation.py` additionally swaps
`_build_fixed_mass_hmc_adapter` → public `build_fixed_mass_hmc_adapter`
(facade exists) — but see Decision 5 for its classification.

### Group C — leave as diagnostic with nonclaims (no code change)

`scripts/run_ccma_broad_fixed_metric_l_epsilon_search.py` (discard-only, test
pins route), `scripts/run_two_currency_double_zlb_dz5_neutra_fixed_metric_grid.py`
(no downstream consumer found), `scripts/ccma_operational_broad_l_epsilon_neighbor_guard.py`
(route self-labels `adaptive_diagnostic_only`), the two one-country ZLB
diagnostic scripts (already canonical or scripted-candidate only), and
`cross_country_multi_asset_bayesfilter_mass_preconditioner.py` (symbol hits
are forbidden-symbol assertions proving non-use).

### Group D — no action (string/key-only hits; already canonical)

`one_country_zlb_ns_estimation.py` (all 28 hits are payload keys; calls
canonical `tune_hmc_kernel` at :9967, :10989, :12462),
`cross_country_multi_asset_macro_mixed_frequency_hmc_kernel_tuning.py`
(enforces `runtime == "bayesfilter.inference.tune_hmc_kernel"` at :1304),
`..._tuned_retained_estimation.py`, `..._phase3_mass_adaptation.py` (public
result attributes), `daily_asset_midas_phase9_...` (canonical import :28),
and the three phase7/8/8b synthetic campaign wrappers (lineage-key tuples).

### Tests (lockstep-only updates)

Per the Group 5 table in the classification report: authority-string tests
move with the phase4/5t migration; route-pinning tests
(`test_run_ccma_broad_fixed_metric_l_epsilon_search.py`,
`test_two_currency_double_zlb_dz5_neutra_fixed_metric_grid.py:16`,
`test_..._canary.py:27`) change only if their runner migrates; negative
guards (uncertainty confirmation :47, owned-client :1041/:1251,
mass-preconditioner :334) are kept as-is.

## 5. Decisions Required Before Group-A Migration (owner boundary)

1. **SDK facade for the bootstrap fixed-mass adapter.** Add public
   `build_bootstrap_fixed_mass_adapter` (+ class re-export) to
   `bayesfilter.inference` preserving the private class-identity string in
   adapter signatures, or defer all bootstrap-adapter import swaps. Touches
   stop condition 5; adding an export is a BayesFilter interface change that
   the handoff did not pre-authorize.
2. **Frozen-lineage policy for migrated tuners.** Migrating the MIDAS robust
   driver and the C2 phase4 ladder changes authority strings/signatures that
   frozen retained artifacts pin. Proposal: freeze existing artifacts as
   historical evidence (no re-issue), migrate the producers, and let the next
   fresh run produce new-lineage artifacts under the canonical route. Confirm.
3. **dsge_hmc retained validation** — is r16 retained validation claim-bearing
   (migrate adapter build to public `build_fixed_mass_hmc_adapter` + facade
   swap) or diagnostic-only as self-labeled (freeze)?
4. **dsge_hmc mass-injection audit** — leave untouched (recommended; the
   private-class-path string is load-bearing in signature recomputation), or
   plan a coordinated identity-preserving change?
5. **MacroFinance DZ5 fixed-metric canary and MIDAS phase10 validation** —
   both have contradictory evidence (nomination-only/explanatory labels vs.
   load-bearing position in an admission lineage). Classify (a) migrate or
   (b) grandfather?

## 6. Execution Order (after decisions)

Per primary-plan Phase 5 sequence, one family at a time, focused tests after
each:

1. MacroFinance import swaps that are unambiguous and unblocked
   (Group B `_mass_artifact_signature` swaps; `_build_fixed_mass_hmc_adapter`
   → public where the facade exists). Verification: focused MacroFinance
   suite + `python -m compileall` on touched files.
2. dsge_hmc import swaps (round380 adapter, repair/campaign/refinement
   scripts). Verification: focused dsge_hmc suite.
3. MIDAS robust driver migration (per Decision 2), then retained-runner
   signature updates, then Step E test (below).
4. C2 phase4+5t atomic migration with its three test files.
5. CCMA owned-client migration with its tests.
6. Step E: add `tests/test_daily_asset_midas_robust_broad_grid_tuning.py`
   covering the five handoff-required points, after the driver's
   post-migration contract is real.
7. Step F full suites; then Step G GPU canary; Step H remains gated on two
   green cross-repo runs.

No numerical behavior changes in any import swap. Any swap that changes an
artifact signature is not a swap — it stops and gets reported.

## 6a. Contract-Gap Finding (2026-08-21): C2 phase4+5t and CCMA owned client

Migration families 4 and 5 are BLOCKED by a genuinely different numerical
contract, per handoff Section 4's rule ("If a truly different numerical
contract is discovered, stop and document it for review"). Documented here;
no code was changed in either family.

The shared contract: **caller-owned candidate campaign under an externally
frozen precomputed mass artifact, with BayesFilter as runtime and
artifact-packaging authority.**

- `..._phase4_step_trajectory.py` tunes step size per predeclared leapfrog
  count under the frozen Phase-3 mass ("No mass adaptation is performed
  here"), via `run_fixed_mass_hmc_tuning_budget_ladder` per candidate, then
  packages evaluations through `orchestrate_generic_hmc_tuning`.
- `cross_country_multi_asset_bayesfilter_owned_hmc_client.py` runs its own
  step/L grid (`_run_hmc_candidate`) under a caller-frozen mass and uses
  `orchestrate_generic_hmc_tuning` three times purely to package
  rehearsal/preliminary/final artifacts.

Why canonical migration is not expressible: `HMCKernelTuningConfig` restricts
`mass_policy` to `{windowed_adaptive, fixed_identity}`
(`hmc_kernel_tuning.py:1647` et al.) and deliberately exposes no caller
mass artifact, candidate grid, leapfrog count, or budget schedule. Migrating
these chains to `tune_hmc_kernel` would replace their phased frozen-mass
design (mass frozen in an earlier phase; step/L tuned under it) with
canonical internal mass adaptation — changing selection rules, budgets,
seeds, and the phase-gating evidence criteria that downstream tests and
frozen artifacts pin. That is a redesign of a claim-bearing evidence chain,
not a caller migration, and it crosses the stop-condition boundary
("a proposed repair changes ... evidence criteria, or campaign budget").

State of the affected campaigns: both are completed frozen campaigns
(C2 phase4/5T artifacts dated 2026-06-18/19 with
`phase5t_attempt_1_authorized: true` in the ladder rerun2 artifact and the
5T ladder handoff on disk). Under the owner's freeze-old-lineage decision,
they are reclassified as **frozen-campaign historical drivers**: readable,
runnable only against their historical routes, and unable to issue new
canonical admission artifacts (the registry already denies artifact
authority to `run_fixed_mass_hmc_tuning_budget_ladder` and
`orchestrate_generic_hmc_tuning`). Their authority-string tests stay
unchanged, consistently pinning the historical route for the frozen chain.

Owner review items for a future canonical redesign (not executed here):

1. Whether the frozen-mass step/L contract should be served by
   `tune_fixed_transport_hmc_kernel` with the mass-latent affine transform
   as the fixed transport (mathematically the frozen-mass latent target has
   identity mass, i.e. an affine fixed-transport problem) — this is a
   hypothesis, `not checked` against the fixed-transport interface's actual
   config surface.
2. Whether a future C2/CCMA campaign collapses its mass phase into canonical
   `tune_hmc_kernel` end-to-end (windowed adaptive), retiring the phased
   frozen-mass design for those chains.
3. Either path requires its own experiment plan and evidence contract; the
   frozen June campaigns remain valid historical evidence regardless.

## 7. Evidence Contract for the Migration Family Gates

- Question: do consumers behave identically through public facades /
  canonical interfaces on the focused surfaces?
- Comparator: pre-migration focused-suite results (MacroFinance 64/64,
  dsge_hmc 50/50) and, for import swaps, byte-identical artifact signatures.
- Promotion criterion per family: focused suite green + no signature drift.
- Veto: any signature/lineage change from a "behavior-preserving" swap; any
  new failure fingerprint not explained by the migration's declared contract
  change (authority strings are declared changes for the phase4/5t family).
- Explanatory only: runtimes, warning counts.
- Not concluded even on pass: posterior correctness, sampler validity,
  GPU readiness, statistical ranking, full-suite health.
- Artifacts: per-family execution notes appended to this file; final result
  memo per handoff Section 12.

## 8. Family Execution Notes

### Family 1+2: SDK facade and behavior-preserving import swaps (2026-08-20/21) — GREEN

- BayesFilter: `hmc_bootstrap.py` gained public
  `build_bootstrap_fixed_mass_adapter` (same-object alias of the private
  builder; adapter-signature class-identity string preserved by construction)
  and the name was added to `bayesfilter/inference/__all__`. Identity checks
  passed (`is`-comparison in tfgpu); route inventory `--check` clean
  (10 discovered/registered, 0 unclassified, 0 stale).
- Semantic-identity proof for the signature swap: both private
  `_mass_artifact_signature` homes (`hmc_kernel_tuning.py:13551`,
  `hmc_budget_ladder.py:2631`) are one-line delegates to public
  `hmc_artifact_identity.mass_artifact_signature`. Verdict: `correct`.
- dsge_hmc swaps (4 files): `rotemberg_round380_neutra.py`,
  `..._mass_artifact_repair.py`, `..._candidate_campaign.py`,
  `..._candidate_refinement.py` — import moved to
  `hmc_artifacts.mass_artifact_signature` under the original local alias.
  Focused suite after: **50/50** (46.8 s, CPU-hidden tfgpu).
- MacroFinance swaps (15 files): 13 CCMA `phase5*/phase10*` diagnostics
  (module-alias call sites -> `hmc_artifacts.mass_artifact_signature` /
  `hmc_bootstrap.build_bootstrap_fixed_mass_adapter`; 15 call sites),
  `daily_asset_midas_eom_bootstrap_hmc_diagnostic.py` and
  `daily_asset_midas_bayesfilter_retained_synthetic_hmc.py` (direct imports,
  underscore aliases kept so `_require_runtime()` `locals()` keys are
  unchanged). Focused suite after: **64/64**. `git diff --check` clean.
- Not touched, per owner decisions: dsge_hmc `retained_validation` and
  `mass_injection_audit` (frozen debt), MacroFinance DZ5 canary and
  `daily_asset_midas_phase10_synthetic_hmc_validation.py` (grandfathered).
  Note: the phase10 public-adapter swap would also not have been
  behavior-identical — public `build_fixed_mass_hmc_adapter` computes the
  mass signature itself instead of accepting the caller's, confirming the
  freeze.

### Family 3 + Step E: MIDAS robust driver migration (2026-08-21) — GREEN

- `daily_asset_midas_robust_broad_grid_tuning.py` migrated from
  `tune_hmc_kernel_robust_broad_grid` to canonical `tune_hmc_kernel`:
  `HMCKernelTuningConfig(seed=ROOT_SEED, target_scope=..., tf_function,
  use_xla=False)`; geometry handoff via `initial_covariance` +
  `parameter_scales`; result processing on `HMCKernelTuningResult`; public
  artifact records `tuning_authority: bayesfilter.inference.tune_hmc_kernel`;
  schemas bumped to `canonical_kernel_tuning_*.v1`; new dated artifact scope
  `daily_asset_midas_canonical_kernel_tuning_2026_08_21`. Frozen 2026-08-14
  robust-grid campaign and the retained runner's pinned signatures untouched;
  lineage boundary documented in the driver docstring. Dropped:
  `MASS_PREPARATION_SEED` and the L-grid (canonical route owns mass windows
  and candidate grids by design).
- Step E test added: `tests/test_daily_asset_midas_robust_broad_grid_tuning.py`
  (AST/source-anchor contract tests, 6 tests — driver has import-time GPU
  guards so it is not importable CPU-hidden). Covers the five
  handoff-required points. Result: **6/6**; focused suite including it:
  **70 passed**.
- Nonclaim: no tuning campaign was run; contract-shape evidence only.

### Families 4-5: BLOCKED — see Section 6a contract-gap finding.
