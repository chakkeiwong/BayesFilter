# BayesFilter HMC Tuning Interface Simplification and Broad-Grid Repair Plan

Date: 2026-09-05
Status: complete for the scoped interface, deterministic implementation, and
documentation work; no numerical or scientific promotion claim
Scope: BayesFilter HMC route identity, ordinary fixed-metric selection,
documentation, examples, and deterministic/focused tests. No long HMC campaign
or scientific promotion is authorized by this plan.

## Decision

BayesFilter will present one recommended tuner for each supported target class.
The target contract, not a caller's preferred grid, selects the tuner.

| Situation | Primary recommendation | Conditional alternative |
|---|---|---|
| Ordinary target with an exact log value and matching exact score in one coordinate system | `tune_hmc_kernel` with `HMCKernelTuningConfig` | Reuse a previously admitted artifact only when its complete tuning scope is unchanged. Otherwise retune. |
| Frozen nonlinear transport with the exact Jacobian-corrected transformed value and matching transformed score | `tune_fixed_transport_hmc_kernel` with the measured joint-grid policy | The legacy directional policy may be used only to debug mechanics; it cannot issue a verified handoff. |
| Deterministic position-only proposal field that is not the exact score, with an exact endpoint target available | `bind_neural_force_hmc_tuning_runner` followed by `tune_hmc_kernel` with `TensorFlowHMCKernelTuningConfig` | None for artifact-authoritative ordinary tuning. This branch is mechanics/candidate evidence only and must not be described as exact-gradient HMC. |
| Chain mechanics smoke or historical replay | The specific runner or stage helper named by the test | None. A runner or stage helper is not a tuner and cannot be promoted by a caller. |
| No exact target contract matching one of the rows above | Stop | Implement and review a new target-specific route; do not assemble helpers into a private tuner. |

There are exactly two active artifact-authoritative public tuner names:
`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`. Compatibility
delegates, chain runners, stage helpers, discovery helpers, and historical
tuners remain classified in the executable registry, but they do not appear as
recommended alternatives.

## Ordinary HMC procedure

The ordinary exact-value/score route has one policy. BayesFilter, not the
consumer, owns all of these steps:

1. Validate the target value/score capability, target scope, coordinates,
   initial position, and optional geometry hint.
2. Construct and screen the bootstrap fixed-mass kernel.
3. Perform windowed mass adaptation and freeze the resulting metric and
   post-warmup four-chain start bank.
4. Evaluate the complete primary leapfrog grid
   `L=(3, 5, 9, 13, 18, 25)`. This six-point grid is an inherited reviewed
   BayesFilter policy promoted by the owner's 2026-09-05 decision. It is a
   coarse coverage design, not a claim that omitted integers are equivalent.
5. Tune epsilon independently for every primary `L`. One epsilon must never be
   shared across different `L` values in the authoritative ordinary route.
6. Preserve every primary result. A candidate-local non-finite failure rejects
   that candidate; a shared target, coordinate, metric, runner, or schema
   failure invalidates the search.
7. Add every untested integer midpoint adjacent to every surviving primary
   candidate, using the single repository-owned refinement function. Tune a
   fresh epsilon independently for every refinement `L` and screen it from the
   same frozen metric and start-bank contract.
8. Use a deterministic ordering only to choose which viable pairs enter the
   bounded fresh verifier first. This ordering is not a statistical ranking or
   a claim that one viable pair is superior.
9. Run disjoint fresh fixed-kernel verification. Emit a final kernel only after
   the verifier passes. All tuning and selection draws remain discarded.

The default upper bound is `L=25`, inherited from the existing fixed-metric
broad-grid policy. The public ordinary config does not expose a different grid
or an anchor-relative alternative. A future proposal to search above 25 is a
new numerical-policy decision and requires a separately reviewed evidence
plan; selecting `L=25` does not by itself prove that a larger value would or
would not work.

The geometry-derived trajectory target remains useful as explanatory telemetry
and deterministic candidate ordering. It does not construct the primary grid,
exclude candidates before measurement, or replace target-specific screening.
The formula is exact for the local quadratic approximation used to derive it;
it is not treated as globally exact for a nonlinear posterior.

## Compatibility disposition

The prior ordinary ID `operational_paired_fixed_trajectory_selection_v3`
describes the shared-epsilon `{floor(anchor/2), anchor, 2*anchor}` procedure. It
must remain readable as historical provenance, but it loses public artifact
authority and is not a selectable public default.

The explicit `joint_l_epsilon_grid_fixed_mass_hmc` route and P4-E
anchor-offset policy remain lower-level diagnostic compatibility mechanisms.
The public `HMCKernelTuningConfig` rejects a non-null
`engineering_probe_covariance_multiplier` before target or HMC work. Internal
stage tests may continue to exercise P4-E, but the guide must not present it as
an ordinary tuning choice.

The broad diagnostic helpers remain callable for historical result replay and
specialized diagnostics. They must share the primary-grid/refinement primitive
with the canonical tuner and remain registry-classified as diagnostic with
replacement `tune_hmc_kernel`. They are not a second recommended route.

## Code trace and single-source requirements

The audit covers every layer that can change the effective procedure:

| Layer | Inspected behavior | Required repair |
|---|---|---|
| Public dispatch, `hmc_tuning_dispatch.py` | One facade dispatches by typed config | Keep one facade; explain that the typed position-field config is a conditional mechanics branch, not a second ordinary authority path. |
| Public definitions and package exports | The pre-repair tree exposed an ordinary implementation definition in addition to its dispatcher, while fixed transport already had one implementation | Retain exactly one function definition per public tuner and test package-export object identity. |
| Capability registry, `tuning_contract.py` | Eight diagnostic or historical procedures are misleadingly typed as `public_tuner` | Give them explicit `diagnostic_helper` or `historical_helper` kinds, bump the capability schemas, and require each to name its active replacement. |
| Route identity, `hmc_route_contract.py` | The v3 paired selector currently owns artifact authority | Introduce a versioned broad fixed-metric ordinary ID, make it the sole ordinary artifact-authority ID, and keep v3 non-authoritative. |
| Public config, `hmc_kernel_tuning.py` | A covariance multiplier silently selects P4-E while retaining the v3 ID | Make broad fixed-metric selection the default and reject the P4-E switch at the public config boundary. |
| Phase 4 mass handoff | Operational warmup freezes an adapted metric and four-chain bank | Reuse that checked bank for all broad candidates; do not create a new start distribution per `L`. |
| Phase 5 selector | v3 uses a shared epsilon; P4-E uses anchor offsets and per-`L` ladders | Replace the authoritative branch with the broad primary barrier and survivor-midpoint refinement, with independent epsilon ladders. |
| Phase 6 | The shared-epsilon branch can perform another local `L` search | Bypass a second `L` search after broad Phase 5; carry measured pairs directly to Phase 7. |
| Phase 7 | Candidate-batch handoff runs fresh verification, but the public config still exposed a verified-log-midpoint switch implemented for the old shared-epsilon selector | Use the direct broad-candidate queue, fix the public compatibility field at `single_repair`, and permit the old midpoint-bracket machinery only when a lower-level caller explicitly selects the historical shared-epsilon algorithm. |
| Work accounting | The old manifest budgeted a three-candidate replicated selector rather than all broad per-`L` ladders | Bind the manifest to the six-primary plus seven-maximum-refinement bound, one ladder per `L`, and direct Phase 7 verification. |
| Fixed-metric diagnostic module | Owns the existing six-point grid and midpoint rule | Move those pure policy values/functions to one dependency-free policy module and import them from both implementations. |
| Robust/operational broad wrappers | Additional diagnostic callable names exist | Keep registry classification and replacement explicit; do not export or document them as recommended tuners. |
| Public ordinary diagnostic helpers | Four legacy tuning diagnostics plus the start-bank diagnostic are package-reachable but were absent from the first executable inventory | Classify all five as diagnostic-only, name `tune_hmc_kernel` as their replacement, and make the AST inventory fail on any future unclassified diagnostic entry. |
| Fixed-transport tuner | Uses its own measured `(epsilon,L)` contract | Leave separate; ordinary mass adaptation and ordinary grid semantics do not transfer. |
| Fixed-transport candidate helpers | Discovery, combined campaign, and refinement are exported but were omitted from the executable route inventory | Register all three as diagnostic-only with `tune_fixed_transport_hmc_kernel` as their sole active replacement; make the inventory discover their names. |
| Typed position-field tuner | Uses a powers-of-two, first-passing mechanics screen with no artifact authority | Keep conditional and explicitly non-authoritative; do not call it the ordinary exact-score procedure. |

The source tree passes the "one recommendation" criterion only when:

* the registry has two and only two active public tuner records;
* `public_tuner` literally classifies only those two active tuners; diagnostic
  and historical procedures have distinct helper kinds;
* only the new broad ordinary algorithm ID has ordinary public artifact
  authority;
* the old shared-epsilon ID and all diagnostic broad IDs fail the artifact
  authority guard;
* the ordinary public config cannot activate P4-E or select a grid;
* the ordinary public config cannot select the historical verified-bracket
  repair route;
* package exports resolve to exactly one implementation definition for each
  public tuner;
* the canonical and diagnostic broad implementations import the same primary
  grid and midpoint function; and
* the guide's decision table maps each supported situation to one primary
  recommendation and at most one explicitly conditional, weaker alternative.

## Evidence contract

Question: does BayesFilter now give an unambiguous, executable tuning procedure
for each supported HMC target contract, with broad-first ordinary selection and
no hidden public policy switch?

Baseline: the pre-repair v3 ordinary path, which shared one epsilon over a local
three-value grid, plus the publicly reachable P4-E multiplier branch.

Primary pass criterion: deterministic route/config tests and focused
stage-fixture tests prove the new route identity, exact primary/refinement
candidate construction, independent per-`L` tuning lineage, direct fresh
verification handoff, and rejection of former public switches.

Promotion vetoes: more than two active public tuner records; an old or
diagnostic route accepted by the artifact guard; a public ordinary config that
can choose a grid or P4-E; duplicate broad-grid definitions; shared epsilon
across different `L`; Phase 6 performing a second `L` search; generated docs
drifting from the registry; or any failing focused test.

Explanatory diagnostics: candidate ordering, geometry trajectory target,
acceptance values, candidate count, and runtime. They do not rank viable
stochastic candidates or establish posterior convergence.

What will not be concluded: these static and fixture checks do not establish
that any `(epsilon,L)` pair works for MacroFinance Phase 14, that the adapted
mass is adequate, that `L<=25` is sufficient for every target, that retained
chains converge, or that HMC is scientifically valid for a downstream model.

Preserved result: this plan, the code/test diff, regenerated route tables, and
focused test output. No numerical campaign artifact is produced.

## Implementation phases

### Phase 1: identity and pure policy

1. Add a dependency-free ordinary broad-selection policy module containing the
   versioned policy ID, primary grid, and survivor-midpoint function.
2. Make the fixed-metric diagnostic module import and re-export those values for
   compatibility instead of defining a second copy.
3. Add the new ordinary algorithm ID to the route contract. Map it to the
   existing operational windowed warmup and grant artifact authority only at
   the top-level selection stage.
4. Retain the v3 paired algorithm as supported historical/diagnostic identity
   with no artifact authority.

### Phase 2: canonical ordinary execution

1. Make `HMCKernelTuningConfig` default to and require the new algorithm ID.
2. Reject `engineering_probe_covariance_multiplier` in the public config with
   a message that identifies it as a lower-level diagnostic compatibility
   mechanism.
3. Route a passed operational mass handoff into the broad Phase 5 path.
4. Execute the full primary barrier, then one midpoint-refinement barrier around
   every primary survivor. Use independent epsilon ladders and independent
   seeds for every `L`.
5. Aggregate candidates across both barriers and preserve all candidate-local
   failures. Deterministically nominate one viable pair without making a
   superiority claim.
6. Hand the complete eligible batch directly to Phase 7 fresh verification;
   never invoke the local shared-epsilon Phase 6 search.
7. Bind the new algorithm and policy IDs, exact grids, mass signature,
   start-bank signature, target scope, and runner identity into diagnostics and
   replay payloads.
8. Fix the public verification-repair compatibility field at `single_repair`.
   Keep `one_verified_log_midpoint` reachable only through an explicitly
   selected historical shared-epsilon lower-level config.
9. Budget the complete broad search: six primary candidates plus at most seven
   midpoint candidates, with one independent epsilon ladder per candidate.

### Phase 3: interface documentation

1. Put the decision table and two active tuner names at the start of the guide.
2. Describe the ordinary nine-step broad-first procedure once. Move diagnostic
   helper details to a clearly labeled compatibility appendix.
3. Describe the typed position-field branch as a conditional mechanics path and
   remove any statement that it provides ordinary artifact authority.
4. Keep the fixed-transport measured-grid procedure separate and state its one
   legacy diagnostic alternative.
5. Update the normative LaTeX chapter, examples, capability registry, generated
   route tables, and documentation tests from the executable policy.
6. Replace the misleading non-active `public_tuner` capability classification
   with explicit diagnostic/historical helper kinds and version the schema.
7. Classify every package-reachable ordinary function whose name advertises a
   tuning diagnostic or start-bank diagnostic. These helpers remain available
   for focused testing, but they are not additional route choices.

### Phase 4: tests and validation

1. Test the pure primary grid and exact midpoint sets, including multiple
   survivors and no-survivor behavior.
2. Test route truth tables: new ID authoritative, v3 and joint IDs supported
   only as non-authoritative compatibility routes.
3. Test public config defaults and fail-closed P4-E/v3 overrides without target
   evaluation.
4. Use stubbed Phase 4/5/7 fixtures to assert the initial six `L` values are all
   attempted before refinement, epsilon is independently tuned, refinement is
   survivor-driven, and Phase 6 is bypassed.
5. Test that the registry still has exactly two active tuner names and that
   diagnostic helpers name `tune_hmc_kernel` as their replacement.
6. Regenerate and check documentation tables; compile/import examples and, when
   available, build the LaTeX document.

## Skeptical pre-execution audit

The original plan failed this audit because it treated the local
shared-epsilon route, P4-E, and broad search as peer choices and left the broad
route diagnostic-only. That contradicted the owner's decision and would have
preserved the interface ambiguity.

The amended plan passes for implementation:

* Wrong baseline: v3 and P4-E are retained only as the behaviors being replaced,
  not as scientific baselines or warm starts.
* Proxy promotion: acceptance and the geometry trajectory target only nominate
  verifier order; neither establishes convergence or superiority.
* Missing stop conditions: route mismatch, shared invalidity, all-candidate
  failure, incomplete primary barrier, failed fresh verification, and test/doc
  drift all fail closed.
* Unfair comparison: every `L` in a barrier receives its own epsilon tune from
  the same frozen metric/start-bank contract and independent seeds.
* Hidden assumptions: the six-point grid, `L=25` cap, one refinement round,
  four-chain bank, and deterministic ordering are named as versioned policy,
  not universal mathematical facts.
* Stale context: the plan follows the current dispatcher, registry, phase
  handoffs, replay guards, typed binding, and fixed-transport implementation,
  not only prior memos.
* Environment mismatch: implementation validation is CPU-safe construction and
  stub-fixture work; no GPU/HMC research run is part of this plan.
* Uninformative commands: every planned test maps to a stated interface or
  stage invariant. No short smoke result will be interpreted as numerical
  tuning evidence.

A second skeptical audit during execution found stale verified-bracket tests
silently following the new broad default. The failure was informative: the
public option was implemented only for the old `operational_selection_v2`
handoff and its three-start accounting did not describe a two-candidate broad
queue followed by bracket repairs. The plan was amended before proceeding.
The repaired boundary fixes the public one-call tuner to one policy and makes
every remaining bracket test opt into the historical shared-epsilon algorithm
by ID. This removes a dead public branch rather than increasing its budget and
preserving a second ordinary procedure.

A final export audit found five package-reachable ordinary diagnostic helpers
that the first AST route inventory did not discover. This was a documentation
and future-migration defect: their names were diagnostic, but the executable
registry did not prove their lack of authority or name the supported
replacement. The plan was amended before completion to classify them and to
make future unclassified diagnostic-style function definitions fail the route
inventory check.

## Completion record

Execution completed on 2026-09-05. The terminal skeptical review passes for the
scope of this plan. The implementation has one authoritative ordinary policy,
one fixed-transport policy, one definition for each public tuner, and no hidden
public selector that can restore P4-E, a caller-selected grid, shared epsilon,
or the historical verified-midpoint repair. This is an interface and
deterministic-control-flow result, not numerical evidence for an HMC kernel.

### Implemented files

The scoped implementation changed these code and public-export files:

* `bayesfilter/hmc_ordinary_selection_policy.py`
* `bayesfilter/hmc_route_contract.py`
* `bayesfilter/hmc_budget_contract.py`
* `bayesfilter/__init__.py`
* `bayesfilter/inference/__init__.py`
* `bayesfilter/inference/hmc_fixed_metric_grid_search.py`
* `bayesfilter/inference/hmc_kernel_tuning.py`
* `bayesfilter/inference/hmc_phase5_evidence_resume.py`
* `bayesfilter/inference/hmc_tuning_dispatch.py`
* `bayesfilter/inference/tuning_contract.py`

The documentation and executable audit surface changed in:

* `docs/reference/hmc-tuning-interface.md`
* `docs/chapters/ch21b_hmc_tuning_interfaces.tex`
* `docs/generated/hmc_tuning_route_table.md`
* `docs/generated/hmc_tuning_route_table.tex`
* `docs/examples/hmc_tuning_ordinary.py`
* `docs/examples/hmc_tuning_neural_force_binding.py`
* `docs/examples/hmc_tuning_route_selection.py`
* `docs/benchmarks/run_pp_ukf_tuning_repair_canary_20260721.py`
* `scripts/inventory_hmc_tuning_routes.py`
* `scripts/audit_ordinary_hmc_migration_surface.py`
* this plan

The focused regression changes are in:

* `tests/test_hmc_ordinary_selection_policy.py`
* `tests/test_hmc_budget_contract.py`
* `tests/test_hmc_kernel_tuning_fixed_mass_step.py`
* `tests/test_hmc_kernel_tuning_frozen_step_trajectory.py`
* `tests/test_hmc_kernel_tuning_outer_loop.py`
* `tests/test_hmc_kernel_tuning_p4_registry_public_chain.py`
* `tests/test_hmc_kernel_tuning_public_api.py`
* `tests/test_hmc_route_contract.py`
* `tests/test_hmc_tuning_contract.py`
* `tests/test_hmc_tuning_dispatch.py`
* `tests/test_hmc_tuning_documentation_contract.py`
* `tests/test_hmc_tuning_gap_regressions.py`
* `tests/test_hmc_tuning_policy_replay_authority.py`

Unrelated dirty-worktree files and generated research artifacts were preserved
and were not made part of this repair.

### Verification evidence

* The final CPU-only 13-file ordinary suite completed with `372 passed`:
  `CUDA_VISIBLE_DEVICES=-1 pytest -q` followed by the 13 test paths listed
  immediately above.
* The fixed-transport regression suite completed with `95 passed`:
  `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_fixed_transport_hmc_tuning.py
  tests/test_fixed_transport_hmc_binding.py
  tests/test_fixed_transport_hmc_candidate_discovery.py
  tests/test_fixed_transport_hmc_grid_policy.py`.
* `CUDA_VISIBLE_DEVICES=-1 pytest -q
  tests/test_ordinary_hmc_migration_audit.py` completed with `3 passed`.
* A final registry, route, dispatcher, and documentation subset completed with
  `63 passed` after the rendered-document repairs. The documentation contract
  was rerun after moving the exhaustive registry behind the procedures and
  completed with `15 passed`.
* `python scripts/inventory_hmc_tuning_routes.py --check` discovered 18
  tuning-related definitions, with exactly two active public tuners, 13
  diagnostic helpers, three historical helpers, no stale registry entries, and
  no unclassified definitions.
* A definition search found exactly
  `hmc_tuning_dispatch.py::tune_hmc_kernel` and
  `fixed_transport_hmc_tuning_tf.py::tune_fixed_transport_hmc_kernel` as the
  two public tuner definitions.
* `python scripts/render_hmc_tuning_interface_docs.py --check`, targeted
  `python -m py_compile`, and `git diff --check` all passed.
* The route-selection and neural-force binding examples executed as
  construction checks. The ordinary example stopped at its intentionally tiny
  contract-only budget; that result was not interpreted as numerical evidence.
* `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` produced the
  552-page document. Rendered pages from the HMC chapter were inspected after
  repair; that chapter has no overfull boxes. Three unresolved citations remain
  elsewhere in the pre-existing monograph (`Afshar2015`, `Gorinova2020`, and
  `Pakman2014`).

### Trace and migration audit

The static downstream audit was written only to
`/tmp/bayesfilter-hmc-interface-audit-2026-09-05`, using:

```text
python scripts/audit_ordinary_hmc_migration_surface.py \
  --downstream-root /home/ubuntu/python/MacroFinance \
  --downstream-root /home/ubuntu/python/dsge_hmc \
  --output-dir /tmp/bayesfilter-hmc-interface-audit-2026-09-05
```

It classified 204 consumer files: 168 in MacroFinance and 36 in `dsge_hmc`.
The roles were 22 public-tuner references, one mixed public/lower-level
reference, 53 raw-runner references, four explicit diagnostic/historical
references, 85 other HMC references, 31 unknown dynamic-import rows, and eight
rows with no relevant resolved reference. Of these, 163 require manual
claim-role review. The source and consumer scan reports 32 unknown dynamic
imports and 20 unresolved dynamic attributes in total. These rows are a
preserved downstream migration queue, not evidence of another BayesFilter
public tuner. The audit also found no unqualified non-XLA default in the scanned
guidance.

### Deliberate compatibility and remaining blockers

The historical shared-epsilon ordinary ID, P4-E internals, broad diagnostic
wrappers, old verified-midpoint repair, and fixed-transport directional policy
remain readable or directly testable only as explicitly non-authoritative
compatibility surfaces. Every discovered tuning-style helper names one of the
two active tuners as its replacement.

Seven ordinary runtime-candidate modules still use NumPy, and the ordinary
runtime still has the documented XLA-default mismatch. The repository-issued
claim-bearing guard therefore remains closed. The downstream manual-review
queue also remains to be migrated in the owning repositories. Finally, this
work did not test a MacroFinance or DSGE target, prove that any
`(epsilon, L)` pair passes, establish that `L<=25` is sufficient for every
nonlinear geometry, validate retained-chain convergence, or support a sampler,
posterior, performance, scientific-validity, or default-readiness claim.
