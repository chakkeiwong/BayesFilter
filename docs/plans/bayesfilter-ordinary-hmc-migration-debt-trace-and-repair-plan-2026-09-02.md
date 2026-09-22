# BayesFilter Ordinary HMC Migration-Debt Trace And Repair Plan

Date: 2026-09-03

Repository: `/home/ubuntu/python/BayesFilter`

Observed source revision: `54201f5cd925ed15036bad8156606b812d53b045`

Status: amended after source verification and bounded plan review. Static and
authority-boundary work is authorized; no HMC, tuning, benchmark, or promotion
run is authorized by this document.

Amendment record (2026-09-03): the prior Claude response was a first-stage,
plan-only coherence review. Its `VERDICT: AGREE` is scoped to that question and
is not a source audit. A subsequent source trace corrected the branch wording:
the ordinary default deterministically enters the operational warm-up and
fixed-trajectory selector, while the explicit legacy joint-grid identifier can
select the alternate branch. The trace also confirmed a P1 public authority
boundary gap: a non-promoting legacy route is accepted by the public ordinary
orchestration unless an explicit authority check is added. This amendment
records those facts and narrows the executable scope to static, construction,
serialization, and guard work until the unresolved numerical policy choices
receive a separate evidence contract.

Fresh Codex review record (2026-09-03): the independent red-team review
returned `VERDICT: REVISE`. It confirmed the branch and legacy-path findings,
and identified four required amendments before execution: the handoff had a
wrong route-contract path; `operational_authority` is a stage-route property,
not by itself public artifact authority; known NumPy use means the ordinary
route cannot issue a claim-bearing artifact until that policy gate is repaired
or the route is explicitly non-admitting; and unresolved dynamic imports,
static-command scope, and future baseline requirements needed fail-closed
rules. Those findings are adjudicated below; the numerical policy remains
blocked.

Source-prose correction (2026-09-03): the ordinary module header and the
`run_hmc_fixed_mass_step_stage` docstring still described the legacy joint grid
as a promoted Phase 5 policy. They now describe the operational default and
the explicit legacy/internal diagnostic branch. A documentation regression
test protects that distinction; this is a wording/trace repair, not a
numerical policy selection.

## Executive Finding

The previously reported frozen-epsilon tuning mistake is part of a wider
migration debt. The public name `tune_hmc_kernel` is active in the route
registry, but it does not identify one executable policy. Its ordinary config
path and its TensorFlow-native config path enter materially different
implementations, with different authority and evidence roles. For the
ordinary `HMCKernelTuningConfig`/`None` path, the traced default is
deterministic: windowed mass warm-up produces an operational warm-up result,
which selects the fixed-trajectory screen using one shared/frozen epsilon,
`floor(anchor/2)`, `anchor`, and bounded `2*anchor`, with exactly three
replications, followed by exact-L epsilon retuning. The explicit
`LEGACY_JOINT_L_EPSILON_ALGORITHM_ID` selects the alternate segmented/joint
grid branch when that route is constructed (and direct stage tests can invoke
it), but it is classified by the route contract as legacy/non-promoting. The
pre-repair public boundary did not fail closed when that non-promoting route
was combined with ordinary authority-producing orchestration; the construction
guard added in this execution now rejects that combination.

The documentation and capability registry correctly state several important
boundaries, including that only two public entry points have artifact
authority and that the TensorFlow-native branch is mechanics-only. They do not,
however, state which config variant and internal policy the ordinary public
default actually selects. The generated route table therefore describes a
capability at a coarser level than the executable default. This was an
authority and migration problem in the pre-repair source; the current working
tree now binds the resolved policy and closes the public legacy path. It is not
evidence that any particular HMC policy is numerically correct.

There are additional confirmed or likely migration debts:

* the same public function dispatches both an ordinary monolith and a
  TensorFlow mechanics handoff;
* `tune_hmc_kernel` has a second compatibility definition in the monolith,
  while the registry and lazy exports point at the dispatcher, so import path
  and signature can change what an agent believes is authoritative;
* the route resolver still accepts a legacy joint-grid algorithm while the
  top-level registry reports only the function-level route;
* diagnostic and historical tuners remain exported and are still imported by
  active-looking MacroFinance and `dsge_hmc` scripts;
* raw full-chain mechanics are widespread and are not machine-classified by
  the route inventory, so a passing inventory check does not prove that
  downstream code used an authoritative tuner;
* defaults and policy constants are duplicated across selector, broad-grid,
  budget-ladder, and generic orchestration modules without one target-scoped
  provenance ledger; and
* the active ordinary config defaults `use_xla=False`, which is outside the
  repository XLA-default policy unless it is explicitly recorded as a reviewed
  non-default exception;
* historical plan prose remains easy to discover and still describes that
  non-XLA default as passed, so an agent can inherit a superseded policy even
  when the code and current governance disagree; and
* NumPy remains in several ordinary tuning/selection execution paths despite
  the repository's diagnostic-only NumPy policy.

The terms `operational_authority`, `artifact_authority`, and scientific or
promotion authority are deliberately distinct. In the current route contract,
`operational_authority=True` identifies an operational stage route, while the
same payload says `evidence_role="engineering_only"` and
`promotion_role="stage_handoff_only"`. It must not be used as a proxy for
permission to issue a claim-bearing public artifact. Until the NumPy/runtime
gate and the artifact schema are repaired, an ordinary result may be retained
as an engineering-only diagnostic handoff, but it is not eligible to support a
claim, default promotion, or posterior admission.

The backend, stale-prose, and NumPy points require a bounded source audit
before their exact scope is promoted from migration debt to a repair
commitment. They must not be silently ignored merely because the current
route-inventory test passes.

## Scope And Non-Scope

This plan covers ordinary HMC tuning interface identity, route ownership,
policy disclosure, artifact lineage, downstream migration, static guards, and
the documentation that instructs future agents. It also records the boundary
between ordinary HMC and the separately repaired fixed-transport tuner.

This plan does not authorize:

* an HMC chain, tuning ladder, grid, or GPU run;
* changing the ordinary numerical default before the policy decision and
  evidence contract are reviewed;
* deleting historical modules or artifacts;
* a public release, external message, package installation, or environment
  mutation; or
* treating an acceptance screen, R-hat, ESS, runtime, or a short chain as a
  correctness or superiority claim.

The dirty worktree at planning time contains unrelated user and agent edits.
All implementation work must preserve those edits and must be performed in
small, reviewable commits.

The observed revision above is `HEAD`, not a clean snapshot. In particular,
`bayesfilter/inference/tuning_contract.py`, the HMC guide, the generated route
table, and the HMC chapter contain uncommitted overlays. Findings about the
already-repaired fixed-transport wording refer to that worktree overlay only;
they are not evidence that the overlay is committed or that it belongs in the
ordinary-HMC repair. Claude must report whether each anchor came from `HEAD`
or the working tree before accepting a conclusion.

## Trace Of The Current Code

The following is the source-level path traced at the observed revision. Line
numbers are anchors for this revision and must be rechecked after edits.

```text
public tune_hmc_kernel
  -> require_active_hmc_tuning_route("tune_hmc_kernel")
  -> dispatch on config type
       -> TensorFlowHMCKernelTuningConfig
            -> repository-issued runner binding
            -> TensorFlow mechanics candidate tuner
            -> candidate handoff; artifact authority is false
       -> ordinary/None config
            -> _run_canonical_hmc_tuning
                 -> geometry initialization and bootstrap
                 -> route decision for the windowed mass stage
                 -> default operational warm-up branch
                      -> operational fixed-mass/step stage
                           -> fixed trajectory candidate selector
                                -> bounded {floor(L0/2), L0, 2 L0}
                                -> one frozen epsilon for screening
                                -> exactly three replications per candidate
                           -> exact-L epsilon retune after nomination
                 -> Phase 7 candidate queue and fresh verification/replay
```

The alternate ordinary branch is important but is not the default branch. When
the operational warm-up result is absent, `run_hmc_fixed_mass_step_stage` can
execute the `joint_l_epsilon_grid_fixed_mass_hmc` loop, which adapts epsilon
for each candidate L. An explicit legacy algorithm mapping (or a direct
internal stage invocation) is what reaches that branch; a successful default
ordinary call does not silently fall through to it. The route resolver and
source names currently classify this as a legacy/non-promoting algorithm,
while the historical stage payload says `promoted_default=True`. That naming
and payload conflict remains an owner/schema decision; the missing public
authority check is now repaired, so an agent must inspect the resolved policy
and authority fields rather than infer policy from an algorithm identifier.

The fixed-transport public tuner is a separate route. Its measured joint grid
and frozen-transport requirements must remain separate from ordinary HMC; a
repair to ordinary dispatch must not merge the two interfaces.

There was also a direct source-documentation contradiction inside the ordinary
module: before this execution, `bayesfilter/inference/hmc_kernel_tuning.py:1-15`
and `run_hmc_fixed_mass_step_stage` called Phase 5 the promoted fixed-mass
joint grid, while the public config defaults to the operational route and the
function selects the operational selector whenever the operational warm-up
result is present. The source prose is now corrected to describe the two named
branches; the numerical policy choice remains an owner decision.

## Route And Consumer Inventory

This table distinguishes the function name, the code path it reaches, and the
evidence role. It is the migration map; the existing function-level route
inventory is not a substitute for it.

| Surface | Reachability | Current policy observed | Current authority | Required disposition |
|---|---|---|---|---|
| `tune_hmc_kernel` with `HMCKernelTuningConfig`/`None` | public dispatch -> ordinary monolith -> operational warm-up by default | windowed mass, shared frozen epsilon for bounded floor/anchor/double L screen, three replications, exact-L retune, fresh Phase 7 verification | registry says active; internal selection payload says private/non-promoting | bind resolved policy ID and reconcile prose before promotion |
| `tune_hmc_kernel` preset choice | `HMCKernelTuningConfig.standard()` is used when config is omitted; `serious()` is opt-in | standard is labelled `moderate_local_diagnostic_only` and uses a non-serious budget; serious uses a dimension-scaled policy | function registry is active, but preset role is not an authority identity | document preset role and reject claim-bearing use of an unqualified standard/diagnostic result |
| `tune_hmc_kernel` with `TensorFlowHMCKernelTuningConfig` | same public name -> typed TensorFlow tuner | explicit budgets, powers-of-two-plus-cap L candidates, TFP defaults, no fresh R-hat admission | mechanics/candidate only; artifact authority false | split name or enforce typed role at the boundary |
| `bayesfilter.inference.hmc_kernel_tuning.tune_hmc_kernel` compatibility delegate | direct module import -> dispatcher delegate; lazy package exports resolve the dispatcher | same call eventually dispatches, but the monolith exposes a separate public-looking definition and narrower annotation | not separately registered; discoverability/typing ambiguity | deprecate or make private, align signature and add import-path guard |
| `joint_l_epsilon_grid_fixed_mass_hmc` | alternate fixed-mass stage when operational warm-up result is absent | per-L epsilon ladders and bounded edge repair | route contract labels legacy/non-promoting; historical stage payload still says promoted default | resolve naming/reachability before any default change |
| ordinary public orchestration with an explicit legacy algorithm ID | pre-repair `_run_canonical_hmc_tuning` accepted a supported legacy decision before Phase 7; current facade rejects it | legacy joint-grid handoff was able to reach direct final-kernel construction before the guard | public artifact-authority guard now rejects the route; private diagnostic constructor remains available | classify downstream legacy callers and retain only explicitly diagnostic construction |
| `tune_fixed_transport_hmc_kernel` | separate public fixed-transport module | measured joint grid in frozen transformed coordinates | active artifact-authority route | preserve; do not use for ordinary position-force targets |
| `run_fixed_mass_hmc_tuning_budget_ladder` | exported diagnostic helper and downstream imports | one fixed L, budget escalation/dual averaging | diagnostic only | classify or migrate claim-adjacent callers |
| `orchestrate_generic_hmc_tuning` / `run_generic_hmc_tuning_orchestration` | exported historical helpers and downstream imports | caller-supplied Cartesian epsilon/L grid and acceptance tie-breaks | historical/diagnostic | remove authority language; retain only explicit diagnostics |
| fixed-metric and operational-broad grids | exported diagnostic helpers and downstream imports | module-local L grids, callback-owned adaptation and screens | diagnostic only | per-file role ledger; no canonical handoff |
| robust broad grid | optional diagnostic tuner; some callers use active-looking metadata | per-L adaptation/repair and qualification settings | diagnostic replacement for public tuner | verify scope, XLA, and metadata before migration |
| `run_full_chain_tfp_hmc`, stage helpers, private adapters | direct mechanics or private imports across downstream repos | caller-chosen fixed transition controls | no tuning authority | classify each consumer; add static bypass guard |
| route inventory/tests/docs | function-name registry and string/freshness tests | does not resolve config variant or consumer role | passes today despite bypasses | add policy-aware construction and consumer scans |

The overlapping BayesFilter implementation family that must be classified in
Phase 0 is concrete, not hypothetical: `hmc_tuning.py` (lower-level policies
and diagnostic helpers), `hmc_budget_ladder.py`, `generic_hmc_tuning.py`,
`hmc_fixed_metric_grid_search.py`, `hmc_operational_broad_grid.py`,
`hmc_robust_broad_grid.py`, and `fixed_trajectory_hmc_tuning_v2.py`. The
package export surfaces in `bayesfilter/inference/__init__.py` and
`bayesfilter/__init__.py` expose several of these names. Their module-level
nonclaims are useful evidence, but an export and a nonclaim do not prevent a
downstream caller from treating the result as authority. Each module and each
consumer therefore needs an explicit role row before deprecation or migration.

## Evidence Anchors And Findings

### P1: Function-level authority hides policy-level divergence

Evidence:

* `bayesfilter/inference/hmc_tuning_dispatch.py:29-84` dispatches one public
  function by config type.
* `bayesfilter/inference/hmc_kernel_tuning.py:6401-6460` gives the ordinary
  config an operational fixed-trajectory algorithm by default.
* `bayesfilter/inference/hmc_kernel_tuning.py:10348-10412` chooses the
  operational fixed-mass path whenever an operational warm-up result exists,
  otherwise it enters the joint-grid path.
* `bayesfilter/inference/hmc_kernel_selection.py:392-415,448-482` defines the
  bounded candidate set and enforces three replications.
* `bayesfilter/inference/hmc_kernel_tuning.py:10882-10937` records the
  operational selection payload as a private diagnostic handoff.
* `bayesfilter/inference/tuning_contract.py:851-910` describes ordinary
  capability as owning joint L and epsilon selection, without naming the
  operational branch.
* `docs/reference/hmc-tuning-interface.md:124-201` describes ownership and
  diagnostic alternatives, but does not bind the prose to a config variant and
  resolved policy ID.

Classification: wrong relative to a reader who interprets the documented
ordinary capability as the default measured joint policy; not evidence that
the operational selector itself is mathematically invalid. The exact intended
ordinary policy is an owner decision.

Repair consequence: every result and every guide entry must identify the
config variant, resolved algorithm/policy ID, and evidence role. A function
name alone is not an authority identity.

### P1: The public dispatch accepts two different evidence contracts

The TensorFlow-native config explicitly requires a runner binding and returns
mechanics/candidate evidence (`hmc_tensorflow_tuning.py:250-529`). The ordinary
config can own mass, epsilon, trajectory selection, and fresh verification.
Both are reachable through `tune_hmc_kernel`, and the route registry records
only the function. This is easy for a downstream agent to misread as "the
TensorFlow path is the canonical tuner" or to pass a mechanics handoff where a
claim-bearing ordinary artifact is required.

Classification: confirmed interface ambiguity; the authority distinction is
documented in places, but it is not enforced or visible at the function-level
route boundary.

Repair consequence: use distinct public names or a typed route record that
includes `config_variant`, `policy_id`, `artifact_authority`, and allowed
consumer role. Until then, fail closed on an authority request made with the
mechanics config.

### P1: Preset role is not the same as route authority

`HMCKernelTuningConfig.standard()` is the implicit config when callers pass
`None` (`hmc_kernel_tuning.py:13280`), and its payload labels the preset
`moderate_local_diagnostic_only` (`:15297-15307`). The public budget factory
marks standard work `standard_public_diagnostic` and `serious_policy=False`
(`:15520-15614`); `serious()` is an explicit opt-in. The route registry still
marks the function active and artifact-authoritative. A caller can therefore
see an active route and a passed result while missing the preset's diagnostic
role and budget class.

Classification: confirmed role ambiguity. It is not proof that a standard
result is unusable as a bounded engineering handoff, but it is wrong to read
the function-level active flag as serious/default-readiness authority.

Repair consequence: bind preset role and budget policy to the result's
authority status, state the required preset for each consumer role, and make
claim-adjacent consumers fail closed on an unqualified or diagnostic preset.

### P1: Legacy algorithm acceptance is broader than the active registry

`bayesfilter/hmc_route_contract.py:17-200` knows both operational and
`joint_l_epsilon_grid_fixed_mass_hmc` IDs. The latter is accepted for relevant
stages but is marked non-promoting, while the top-level registry still exposes
one active ordinary route. A caller can therefore construct a route that is
reachable through the active public name but is not an artifact-authority
route, without an obvious public error.

Classification: confirmed boundary gap. Whether the legacy route should be
removed from public construction or exposed under an explicitly diagnostic
name is a policy choice, not a reason to silently change it.

Repair consequence: make stage and public route resolution reject an
authority-producing combination unless the policy record explicitly permits
it; retain a separate diagnostic constructor for historical comparison.

### P1: A non-promoting legacy route can reach the public final-kernel path

The route resolver marks the legacy joint-grid identifier as supported but
non-promoting (`bayesfilter/hmc_route_contract.py:118-200`). The public
ordinary orchestration currently calls `require_hmc_algorithm_route` in
`_run_canonical_hmc_tuning` (`bayesfilter/inference/hmc_kernel_tuning.py:13255-13328`),
which checks support but does not require `operational_authority`. The Phase 7
queue then accepts either an operational-selection payload or a
`joint_l_epsilon_grid_fixed_mass_hmc` handoff
(`hmc_kernel_tuning.py:19925-19957`), and
`_phase7_direct_final_kernel_payload` accepts those source kinds
(`hmc_kernel_tuning.py:26654+`). The resulting object can therefore look like
a final ordinary kernel even though the selected route is explicitly
non-promoting. This is a statically confirmed authority-boundary defect; no
successful HMC run is needed to establish it, and no runtime target claim is
made here.

Classification: wrong relative to the stated authority target. The legacy
stage may remain a useful diagnostic, but it must not issue an ordinary
authority-bearing final payload.

Repair consequence: add a repository-owned operational-authority check at the
public ordinary boundary, add a construction-only regression that a legacy
public request fails closed, and ensure direct legacy diagnostics carry an
explicit non-promoting role. Add a final-payload invariant that rejects a
legacy source kind when an authority artifact is requested.

### P1: Operational route authority is not artifact or scientific authority

`resolve_hmc_algorithm_route` currently reports `operational_authority=True`
for the operational IDs while also reporting `evidence_role="engineering_only"`
and `promotion_role="stage_handoff_only"`
(`bayesfilter/hmc_route_contract.py:157-200`). The ordinary final-kernel and
Phase 7 payloads are private/replay handoffs
(`bayesfilter/inference/hmc_kernel_tuning.py:10920-10937,26654+`), and the
route registry's `artifact_authority` field is a separate function-level
property. Treating `operational_authority` as sufficient for a public
claim-bearing artifact would therefore widen authority without evidence.

Classification: unsupported relative to a claim-bearing artifact target. The
operational route can remain an engineering stage handoff; no scientific,
posterior, default-readiness, or promotion authority follows from that fact.

Repair consequence: define and serialize an explicit `artifact_authority`
decision (and, separately, any promotion/claim role) at the public boundary.
Require all three roles to agree before an authority artifact is issued, and
test that operational route resolution alone cannot grant artifact authority.
Until the known NumPy/runtime policy debt is repaired, the ordinary route must
remain explicitly non-admitting for claim-bearing artifacts.

### P1: Downstream code still calls diagnostic or private surfaces

The route registry labels the following families diagnostic or historical, but
source scans found active-looking consumers in both downstream repositories:

* `orchestrate_generic_hmc_tuning` in
  `MacroFinance/mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase4_step_trajectory.py`
  and `MacroFinance/cross_country_multi_asset_bayesfilter_owned_hmc_client.py`;
* `run_fixed_mass_hmc_tuning_budget_ladder` in the same MacroFinance phase
  scripts and in `dsge_hmc/scripts/run_rotemberg_round380_r16_tuning_repair.py`;
* fixed-metric and operational-broad grid helpers in MacroFinance and
  `dsge_hmc/scripts/run_bgs_bayesfilter_phase08_stage_c_grid_tuning.py`;
* private adapter and mass-signature helpers in several `dsge_hmc` and
  MacroFinance scripts; and
* raw `FullChainHMCConfig`/`run_full_chain_tfp_hmc` mechanics in many scripts,
  including BGS and Rotemberg paths.

The search set includes historical and smoke files, so the count alone is not
an authority claim. The confirmed debt is that the current BayesFilter route
inventory does not classify each downstream consumer before it is used near a
tuning or estimation decision. One MacroFinance module even states that
BayesFilter owns tuning while directly invoking generic orchestration.

Classification: confirmed migration and classification gap; individual files
remain `not checked` until their execution role is read.

Repair consequence: classify every consumer as claim-bearing, candidate,
mechanics-only, smoke/reference, or historical, then migrate only the first
two categories to an approved public route. Keep raw mechanics for declared
diagnostics with explicit nonclaims.

### P2: The route inventory is necessary but not sufficient

`scripts/inventory_hmc_tuning_routes.py:52-113` discovers top-level names and
explicitly excludes full-chain mechanics, internal stages, NeuTra, and
neural-force modules. `--check` can therefore pass while a consumer imports a
private helper, constructs a raw HMC config, or selects a different config
variant. Existing tests assert route counts and documentation wiring, but not
the resolved executable policy.

Classification: confirmed guard coverage gap.

Repair consequence: retain the existing inventory and add a separate static
consumer/bypass inventory. Do not make the first scanner pretend that all HMC
mechanics are tuners; require an explicit classification file for excluded
families.

### P2: Defaults are duplicated and their provenance is incomplete

Observed constants include the floor/anchor/double candidate rule, three
replications, four chains/four blocks/minimum sixteen draws and a 90 percent
t screen in `hmc_verification.py`, fixed metric grids, robust-grid dual
averaging and qualification budgets, powers-of-two TensorFlow candidates,
TFP defaults, and divergent `use_xla` defaults. These are source observations,
not approved new defaults. The prior tuning-function audit and streamline
result already record overlapping families and deferred consumer migration.

Classification: confirmed duplication; provenance and target-scope status are
`not checked` for several values.

Repair consequence: create one policy/assumption ledger. Each value must be
labelled measured, derived, inherited, convenience, reviewed, placeholder, or
diagnostic-only, with an early check and a nonclaim. No numeric value becomes a
new default by repetition.

### P2: NumPy and export surface create additional migration pressure

The ordinary selection/tuning family still imports NumPy in paths identified
by the earlier function audit, while `AGENTS.md:365-395` permits NumPy only in
explicit diagnostic/reference code. `bayesfilter/inference/__init__.py` and
`bayesfilter/__init__.py` export generic, budget-ladder, stage, and broad-grid
helpers alongside public tuners. Existing exports are not proof of authority,
but they make accidental imports likely.

Classification: migration debt under the repository policy; exact runtime
reachability and safe removal order require source tracing.

Repair consequence: first mark exports with explicit diagnostic status and
warn/fail on claim-bearing imports; then perform a separately budgeted NumPy
migration after route identity is repaired. Do not combine a large numerical
rewrite with the authority-boundary change.

### P1/P2: Execution-backend default is not reconciled with repository policy

`HMCKernelTuningConfig.use_xla` defaults to `False`
(`bayesfilter/inference/hmc_kernel_tuning.py:6401-6460`), and the ordinary
dispatcher forwards that value into route validation and every lower-level
runner. `AGENTS.md:420-438` requires XLA JIT by default for algorithmic and
production-target TensorFlow paths; non-XLA is allowed only as an explicitly
classified reference, smoke, debugging, or reviewed exception. The active
function-level route and its standard preset do not currently carry such an
exception identifier. This is a policy-boundary defect even before any timing
or numerical comparison is attempted.

Classification: confirmed default-policy mismatch; whether compatibility,
target coverage, or XLA qualification justifies an exception is an owner
decision. It must not be hidden behind the generic phrase ``use_xla``.

Repair consequence: make the active policy default XLA-on after compatibility
checks, or require an explicit non-default exception role and record its
reason, scope, and nonclaims. Add construction-time tests that reject
`use_xla=False` for an authority role without that exception.

### P2: Duplicate public definition creates an import-path bypass

`bayesfilter/inference/hmc_kernel_tuning.py:14461-14500` defines a
compatibility `tune_hmc_kernel` that delegates to
`hmc_tuning_dispatch.tune_hmc_kernel`, while the route registry and lazy export
maps point at the dispatcher (`hmc_tuning_dispatch.py:29-84`,
`inference/__init__.py:595-606`, `bayesfilter/__init__.py:827`). The delegate
currently preserves execution semantics, so this is not evidence of a second
algorithm. It is nevertheless a confirmed API/discoverability debt: direct
module imports expose a different source location and narrower annotation, and
the existing top-level route inventory explicitly excludes compatibility aliases.

Classification: confirmed import-path ambiguity; numerical behavior is not
checked by this finding.

Repair consequence: retain one canonical public definition, mark the delegate
as private/deprecated with an explicit migration window, align its typed
signature, and add an import-path test/scanner so new consumers cannot treat it
as an independent authority route.

### P2: Superseded policy prose can reintroduce the same default

`docs/plans/bayesfilter_hmc_kernel_tuning_xla_parameter_repair_result_2026_06_22.md:19-99`
records ``Default non-XLA behavior`` as passed because the configs defaulted to
`use_xla=False`. The current governance policy in `AGENTS.md:420-438` says
algorithmic and production-target TensorFlow paths default to XLA and permits
non-XLA only as an explicitly classified exception. The June result may remain
historical evidence of parameter propagation, but it is not a current default
or authority contract.

Classification: confirmed stale-documentation risk; the historical result is
not numerically invalid merely because policy changed, but its default wording
is wrong relative to the current policy if read as guidance.

Repair consequence: add a supersession banner or migration note, distinguish
parameter plumbing from default selection, and scan active guides/examples for
the same unqualified wording. Historical files must remain readable without
being presented as active instructions.

## Research Intent Ledger For Later Numerical Work

This ledger governs a future numerical phase; it is included now so that a
static repair cannot silently turn into an experiment.

| Field | Predeclared statement |
|---|---|
| Main question | Does the repository-owned ordinary tuner select and replay a target-scoped `(epsilon, L, M)` policy under the same evidence contract used by downstream estimation? |
| Candidate/mechanism | First candidate: an explicitly identified ordinary policy. A measured joint `(epsilon, L)` grid is the working migration proposal; operational fixed-epsilon selection and per-L adaptation remain comparison arms until reviewed. |
| Expected failure | A candidate may fail because of implementation, tuning, target/coordinate mismatch, insufficient budget, or sampler geometry. A failure does not by itself reject HMC or the downstream model. |
| Promotion criterion | Exact route/policy identity, finite and replayable artifacts, fresh verification, and predeclared posterior/reference checks appropriate to the target. Numerical thresholds must be justified before the run. |
| Promotion veto | Divergence/non-finite state, route or artifact identity mismatch, missing fresh retuning, failed invariant, invalid target/gradient, missing required diagnostic, or an unrecorded policy branch. |
| Continuation veto | Corrupted target/data, invalid scientific question, changed scope/data/method, exhausted authorized budget, or a stated safety/resource boundary. A failed candidate is not automatically a continuation veto. |
| Repair trigger | Any localized implementation, harness, serialization, policy-lineage, or consumer-classification failure whose scientific scope and budget remain unchanged. |
| Explanatory diagnostics | Acceptance, energy error, path return, ESJD/gradient cost, R-hat, ESS, runtime, and candidate-level summaries unless promoted by an explicit uncertainty design. |
| Must not conclude | No short run or acceptance screen establishes posterior correctness, convergence, superiority, production readiness, source faithfulness, or a new default. |

## Evidence Contract For This Static Phase

* Question: can a reader and a downstream caller determine the exact ordinary
  policy and evidence role from the public route and artifact alone?
* Baseline: the current source at the observed revision, including the
  operational default and all currently exported diagnostic families.
* Primary criterion: a deterministic static trace maps every claim-adjacent
  call to one approved public route, config variant, policy ID, and replay
  schema, with prose and generated tables agreeing with that mapping.
* Hard vetoes: unresolved dispatch branch, authority granted to a mechanics or
  diagnostic route, stale/missing policy identity, an unclassified downstream
  caller, docs that state a different executable default, or a failing static
  regression test.
* Explanatory diagnostics: symbol counts, import graphs, line anchors,
  existing acceptance policies, and NumPy import locations.
* Nonclaims: the static phase does not compare samplers, tune epsilon/L,
  estimate posterior quantities, or rank candidates.
* Preserving artifacts: this plan, the skeptical review, the Claude handoff,
  a machine-readable consumer classification report, route-policy contract
  tests, and a final migration result note with the exact commit and commands.

## Default And Assumption Audit

The values below are observations that must not be promoted without a policy
record. The early checks are static or construction-only unless a later plan
explicitly authorizes a numerical run.

| Choice | Provenance at trace time | Failure mode | Earliest check | Status |
|---|---|---|---|---|
| `{floor(L0/2), L0, 2L0}` | hard-coded source rule, `hmc_kernel_selection.py:392-415` | misses useful L values or gives a false impression of joint tuning | construct a policy payload and assert the candidate set is disclosed | diagnostic/inherited, not a default recommendation |
| exactly three replications | constructor invariant, `:448-482` | weak stochastic screen or accidental all-pass interpretation | schema test plus uncertainty-role assertion | screen rule, not convergence evidence |
| four chains, four blocks, minimum sixteen draws | acceptance-policy source, `hmc_verification.py:452+` | unstable t-screen or overinterpretation with few observations | role test that rejects ranking/convergence claims | diagnostic screen |
| target 0.70 and practical/repair bands | source config/policy | unsupported transfer across targets/scopes | provenance ledger and target-scope check | inherited hypothesis |
| powers-of-two or fixed broad L grids | module-local constants | different consumers compare different candidate spaces | static grid inventory | diagnostic/historical until reviewed |
| TFP dual-averaging defaults | library/inherited behavior | hidden adaptation differences | payload records adaptation parameters and source version | inherited, not self-justifying |
| `use_xla=False` on public ordinary config versus `True` in a broad grid | divergent source defaults | backend/performance comparison is unfair or undocumented | config-construction contract test | unresolved implementation policy |
| stateless seed offsets and start banks | helper/source conventions | candidate order or bank mismatch changes evidence | lineage signature test | must be bound to artifact |
| mass changes after epsilon/L selection | stage sequencing | selected step size is invalid for realized geometry | replay invalidation test | hard lineage requirement |

## Detailed Repair Program

### Phase 0: Freeze the trace and classify the current surface

1. Record the source revision, dirty-worktree boundary, Python/TFP versions,
   and the exact read-only commands used for the trace.
2. Re-read the route registry, dispatch, ordinary monolith, selector,
   TensorFlow tuner, verification policy, generated table, and guide at one
   revision. Resolve line anchors in the resulting report.
3. Build a machine-readable inventory with rows for public functions, config
   variants, internal policy IDs, exports, private imports, direct raw runners,
   and downstream consumers in MacroFinance and `dsge_hmc`.
4. Add an explicit branch-reachability table: default ordinary construction,
   explicit operational construction, explicit legacy construction, and direct
   internal stage tests. Record the resolver decision, stage reached, payload
   source kind, and whether an authority request is permitted for each row.
5. For each downstream file, classify execution role and claim status from its
   calls, metadata, and evidence use. A file is not migrated merely because
   its filename says `tuning` or `repair`; filename or age is never a role
   classifier. Apply the following decision procedure: a file that produces an
   artifact used for a posterior/default/production claim is claim-bearing; a
   file that compares methods under a declared evidence contract is candidate;
   a file that only calls raw transitions or stage helpers with explicit
   nonclaims is mechanics-only; a file explicitly limited to smoke, reference,
   test, or diagnostics is smoke/reference; and a preserved historical reader
   is historical. Multiple roles must be represented as separate rows. If the
   evidence is ambiguous, classify the row as claim-adjacent and require manual
   owner review rather than inferring from a filename, directory, or date.
6. Resolve constant-string dynamic imports with AST and manual call-site
   inspection. For computed `importlib`, `getattr`, entry-point, or other
   unresolved indirection, emit an `unknown_dynamic_import` row and block any
   claim-adjacent admission until a reviewer classifies the target manually.
   Do not use runtime HMC instrumentation to classify a static consumer, and do
   not require a representative numerical run in this phase.
7. Produce an exact NumPy import/call-chain inventory, distinguishing admitted
   runtime use from diagnostic/reference use, before promising a migration.
   At minimum, record the seven currently identified ordinary-family modules
   and their import anchors (`hmc_kernel_tuning.py:35`,
   `hmc_kernel_selection.py:14`, `hmc_tuning.py:10`,
   `hmc_budget_ladder.py:19`, `generic_hmc_tuning.py:15`,
   `fixed_trajectory_hmc_tuning_v2.py:16`, and `hmc_verification.py:13`),
   then trace whether each call is reachable from an admitted public route.
8. Scan active guides, examples, and plans for policy phrases that contradict
   the current XLA/NumPy defaults; mark historical records rather than deleting
   them. Run this stale-policy scan before implementation so it can constrain
   the docs changes.

Deliverable: a versioned, bounded output directory under
`docs/plans/artifacts/ordinary-hmc-migration-debt-2026-09-03/` containing
`ordinary_hmc_surface_inventory.json`, a branch-reachability table, a
consumer-role ledger, a NumPy call-chain ledger, a provenance capture, and a
Markdown trace note. The scan is limited to the named BayesFilter paths and
the explicitly listed downstream files/directories, excludes generated
artifacts and virtual environments, and records the exact file list and
commands. These are classification artifacts, not numerical evidence.

### Phase 1: Establish an unambiguous authority boundary

1. Introduce repository-owned policy records that bind public name,
   `config_variant`, preset role, resolved algorithm ID, policy version,
   `operational_authority`, `artifact_authority`, scientific/promotion role,
   allowed evidence role, and required runner binding. `operational_authority`
   is a stage-route property; it is never sufficient by itself for artifact or
   scientific authority.
2. Make `tune_hmc_kernel` either a single ordinary authority route or a thin
   compatibility dispatcher whose typed variants have distinct names and
   status. The mechanics-only TensorFlow branch must never return an
   authority-bearing ordinary artifact.
3. At the public ordinary boundary, reject an authority-producing request when
   the route resolver returns a non-promoting/legacy algorithm or otherwise
   lacks `operational_authority`; also require an independent
   `artifact_authority` decision and an allowed scientific/promotion role.
   Preserve a separate explicitly diagnostic constructor for historical
   comparisons, and mark its payload `non_promoting` at construction time.
4. Reject or explicitly downgrade a standard/diagnostic preset when a caller
   requests a claim-adjacent or serious consumer role; a function-level active
   flag is not sufficient.
5. Ensure the default config, route decision, stage payload, final-kernel
   payload, and capability registry cannot disagree about the selected policy.
   A final payload must reject a legacy/non-promoting source kind when an
   authority artifact is requested. Because the ordinary runtime currently
   contains known NumPy use forbidden by the repository policy, keep its
   claim-bearing artifact role disabled (engineering-only diagnostics remain
   possible) until the bounded NumPy repair or an explicitly reviewed
   exception is complete. Do not change the numerical policy in this phase
   unless the owner decision in Phase 2 has already been recorded.

Deliverable: route-policy schema with separate operational/artifact/scientific
roles, fail-closed tests, a NumPy-policy admission result, and a compatibility
note.

### Phase 2: Choose and implement the ordinary policy deliberately

The current working proposal is an explicit measured joint `(epsilon, L)`
grid with per-candidate adaptation/verification, because it makes the two
controls visible and prevents a shared frozen epsilon from being mistaken for
joint tuning. This is a proposal, not an approved new default. The owner must
choose among it, per-L epsilon adaptation, or a reviewed dynamic trajectory
policy after reading the evidence contract and cost implications.

If the owner selects the joint-grid proposal, Phase 2 must first decide whether
`joint_l_epsilon_grid_fixed_mass_hmc` is promoted as-is, repaired under a new
algorithm/policy identifier, or retained as a diagnostic comparison. Record
the backward-compatibility impact and update the route registry, payload schema,
and construction tests before any numerical validation. Do not treat the
proposal's clearer control vocabulary as evidence that its implementation is
already the desired default.

For the selected policy:

1. Define the candidate space, adaptation schedule, mass policy, start-bank
   policy, seeds, target/coordinate identity, and stop/repair rules in one
   target-scoped config.
2. Keep calibration, selection, fresh verification, and untouched claim data
   disjoint. Freeze `(epsilon, L, M)` before the claim run.
3. Predeclare the primary selection criterion and its uncertainty treatment.
   Acceptance is a screen or explanatory diagnostic unless the plan gives it
   a statistically defensible role. Do not rank viable stochastic candidates
   from descriptive means alone.
4. Record all failed candidates and repairs without overwriting prior output.
   A failed candidate remains evidence about that candidate, not a reason to
   tune on claim data or transfer settings.

Deliverable: a separate numerical experiment plan and implementation only
after Phase 1 tests pass and the owner resolves the policy choice.

### Phase 3: Make artifacts and replay bind the actual policy

1. Add resolved policy ID, config variant, algorithm route, policy version,
   operational/artifact/scientific authority roles,
   candidate pair, realized epsilon/L, mass signature, target/coordinate
   signature, start-bank signature, adaptation settings, seed lineage, and
   source dependency closure to the result payload.
2. Make public artifact-authority replay and continuation reject missing, stale,
   caller-stamped, mismatched, or cross-model policy artifacts. The existing
   private legacy-schema migration view remains readable historical/diagnostic
   evidence only; it must not be grandfathered into public authority and must
   not be upgraded by adding caller-supplied fields. An old payload lacking a
   resolved policy ID, config variant, route version, or authority roles is
   loadable only as historical/diagnostic evidence; authority-producing replay
   must fail closed and require fresh tuning under the current schema.
3. Invalidate a selected epsilon/L whenever mass, target, coordinates,
   adapter, start bank, backend, dtype, or other bound scope field changes.
4. Preserve old payloads as historical readable evidence; do not silently
   upgrade them by adding a caller-supplied field.

Deliverable: schema version, serializer/replay tests, and migration guidance.

### Phase 4: Migrate downstream consumers

1. Start with the two files that currently label generic orchestration or the
   budget ladder as tuning authority: the MacroFinance phase-4 trajectory
   script and the cross-country BayesFilter-owned client.
2. Maintain a call-site ledger with the exact import, call, config variant,
   artifact role, and classification evidence for every candidate consumer.
   Resolve constant-string dynamic imports statically or by manual inspection;
   do not launch representative HMC runs merely to infer role.
3. Migrate `dsge_hmc` Rotemberg/BGS callers after their roles are classified.
   Replace private adapter/mass-signature imports with repository-owned public
   bindings or a documented mechanics-only interface.
4. Keep historical, smoke, reference, and debugging scripts runnable where
   useful, but add explicit role metadata and nonclaims. They must not emit
   artifacts that look like ordinary authority results.
5. Remove or deprecate broad-grid, generic, budget-ladder, and v2 exports only
   after consumer scans are clean and historical readers are preserved.

Deliverable: consumer classification ledger, migrated call sites, deprecation
messages, and a zero-unclassified-claim-adjacent-consumer report.

### Phase 5: Add static and construction-time guards

1. Extend route inventory to inspect config variants and resolved policy IDs,
   not just top-level function names.
2. Add an AST scanner for downstream imports/calls to private helpers,
   diagnostic tuners, raw full-chain runners, and unclassified HMC configs.
3. Add tests that construct each route without running HMC and assert authority,
   evidence role, policy identity, and forbidden options.
4. Add replay tests for every invalidation field and tests that reject a
   TensorFlow mechanics handoff as an ordinary authority artifact.
5. Add a docs semantic test that compares the guide's stated ordinary default
   to the executable route decision, rather than only checking strings and
   generated-file freshness. Treat construction/replay/serialization success
   as engineering prerequisites, never as promotion evidence.
6. Verify that `docs/main.tex` includes the repaired chapter in the intended
   order and inspect the rendered section; inclusion alone is not evidence
   that the reader-facing explanation is complete.
7. Add an import-path and backend-policy test: the compatibility delegate cannot
   create a distinct route, and an authority result cannot silently use
   `use_xla=False` without a reviewed exception record.
8. Make the test dependencies explicit: route/policy construction precedes
   artifact serialization and docs rendering; the docs semantic check consumes
   structured route data rather than phrase-only matches.

Construction/import-path tests (items 1-3 and 7) may run after Phase 1.
Artifact replay/invalidation tests (items 4-5) require the Phase 3 schema.
The documentation semantic and rendered-section checks require the Phase 6
guidance and generated table. This ordering prevents a partial schema from
being mistaken for a complete guard.

Deliverable: focused pytest suite and a CI/local command that fails closed on
new bypasses.

### Phase 6: Repair the reader-facing guidance

1. State the ordinary default's exact config variant and policy ID near the
   first public example.
2. Put a compact "which route did I call?" decision table in the guide, with
   authority and nonclaim status for every typed variant and diagnostic helper.
3. Explain the old operational selector, the joint-grid alternative, and the
   fixed-transport route as distinct policies. Do not use "joint tuning" unless
   epsilon is actually selected jointly with L under the stated procedure.
4. Include a copyable construction-only inspection command that prints the
   resolved route/policy without launching HMC.
5. Link downstream migration rules and the consumer scanner from `AGENTS.md`
   or the project guide, keeping workflow language out of the monograph prose.
6. Build/inspect the affected `docs/main.tex` section and compare it with the
   protected pre-repair chapter for removed equations, assumptions, citations,
   and qualifications.
7. Add a stale-policy scan that distinguishes historical result notes from
   active guidance and fails on an unqualified non-XLA default in the latter.
   The active check must be tied to structured route/policy data and a
   historical banner, not only to the presence or absence of a phrase.

Deliverable: updated guide, generated table, chapter references, examples, and
documentation-contract tests.

### Phase 7: Separate NumPy and export cleanup

After authority migration is stable, audit each NumPy import and export. Move
diagnostic/reference code to explicit diagnostic modules, replace ordinary
runtime numerical operations with TensorFlow/TFP, and preserve independent
reference tests. This phase gets its own bounded plan and does not change
scientific conclusions by itself.

### Phase 8: Numerical validation (future plan only)

Write a new experiment plan after Phases 1-6. It must include a pre-mortem,
budget, disjoint data partitions, multiple seeds/replications, uncertainty
analysis, exact environment and hardware, versioned output roots, and a
decision table. If the numerical work compares methods, the plan must state
the applicable baseline ladder (naive baseline, best tuned classical baseline,
plain proposed method, and enhanced proposed method), or explicitly justify
why a rung is inapplicable. The first run may use
construction/replay/serialization checks as engineering prerequisites,
followed by the smallest discriminating numerical diagnostic. Those checks are
not promotion evidence. No current artifact may be used as evidence for a new
default merely because it predates the repair.

## Static Verification Commands

These commands are allowed for this documentation and source-trace phase. The
listed pytest and renderer invocations are permitted only after their source
inspection confirms construction-only behavior and no TensorFlow device/HMC
runner initialization; otherwise replace them with static checks. No command
here may launch HMC.

```bash
python scripts/inventory_hmc_tuning_routes.py --check
pytest -q tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py
python scripts/render_hmc_tuning_interface_docs.py --check
awk 'length($0) > 0 && $0 ~ /[[:blank:]]$/ {print FILENAME ":" FNR; bad=1} END {exit bad}' \
  docs/plans/bayesfilter-ordinary-hmc-migration-debt-trace-and-repair-plan-2026-09-02.md \
  docs/plans/bayesfilter-ordinary-hmc-migration-debt-plan-review-2026-09-02.md \
  docs/plans/bayesfilter-ordinary-hmc-migration-debt-claude-audit-handoff-2026-09-02.md
python -m compileall -q bayesfilter/inference scripts/inventory_hmc_tuning_routes.py
python scripts/audit_ordinary_hmc_migration_surface.py \
  --downstream-root /home/ubuntu/python/MacroFinance \
  --downstream-root /home/ubuntu/python/dsge_hmc
git rev-parse HEAD
git status --short --untracked-files=all
rg -n "tune_hmc_kernel|TensorFlowHMCKernelTuningConfig|orchestrate_generic_hmc_tuning|run_fixed_mass_hmc_tuning_budget_ladder|run_fixed_metric_grid_search|run_full_chain_tfp_hmc|FullChainHMCConfig" \
  /home/ubuntu/python/MacroFinance /home/ubuntu/python/dsge_hmc
```

The final `rg` output is an inventory input, not a pass/fail metric. The
external scan is bounded to the named repositories and the file list recorded
in the Phase 0 artifact; it must not recurse into virtual environments,
generated artifact roots, or unrelated repositories. A future AST scanner must
classify each match before it can produce a gate, and must emit
`unknown_dynamic_import` for non-constant indirection rather than silently
passing it. The current scanner's normal report is expected to complete even
when it finds unresolved rows; its `--check` mode is deliberately fail-closed
until both `unknown_dynamic_import` and `unresolved_dynamic_attribute` rows
have been manually classified.

## Stop Conditions And Decision Boundaries

Stop static repair and report a blocker if source inspection cannot determine
which policy produced an artifact, if a route can grant authority to a
diagnostic branch, if a consumer's scientific role cannot be classified, or
if a proposed policy change would alter target/data/backend/scope without an
owner decision. Do not stop merely because a diagnostic candidate fails; that
is a repair trigger under the future numerical plan.

The following choices require explicit owner resolution before numerical
implementation: the ordinary canonical policy, whether to retain one typed
public name or split names, the primary efficiency/selection criterion, and
the campaign budget. This document recommends measured joint `(epsilon, L)`
selection as the clearest migration target, but does not silently authorize
it.

## Acceptance Criteria For This Plan

The plan is ready for implementation only when an independent review confirms:

1. the current default and every alternate branch are traced correctly;
2. authority, candidate, diagnostic, mechanics, and historical roles are
   unambiguous at both API and artifact levels;
3. no proposed metric is silently used as a promotion or continuation gate;
4. stop conditions, repair triggers, scope bindings, and artifact locations
   answer the stated question;
5. a non-promoting/legacy route cannot reach an authority-bearing public
   final-kernel payload, and the regression is construction-only;
6. every claim-adjacent downstream consumer has an evidence-based role row,
   with constant-string dynamic imports resolved statically or manually;
7. downstream consumer migration is bounded and does not delete evidence; and
8. the static tests can fail when prose, registry, dispatch, artifact identity,
   backend policy, or consumer classification drifts.

The accompanying skeptical review records whether these conditions currently
hold. The accompanying Claude memo requests a read-only, exact-path audit of
the plan and source before any code or numerical work begins.
