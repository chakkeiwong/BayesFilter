# BayesFilter Tuning Streamline and Refactor Plan

Date: 2026-08-16; revised 2026-08-17 after Fable R2, posterior-oracle review,
and the Fable coverage-audit response
Status: proposed after source audit; no implementation changes are authorized
by this document alone.
Companion audit: `docs/audits/bayesfilter-tuning-function-audit-2026-08-16.md`

Adjacent execution/evidence authorities are listed in
`docs/audits/bayesfilter-tuning-adjacent-authority-inventory-2026-08-16.md`.
The 26-file/1,417-definition count is the core tuning surface, not the whole
HMC dependency closure. Before extraction, review both the 482 named adjacent
execution/evidence definitions and the 271 additional direct-dependency
definitions listed in the dependency addendum; neither set is optional.

## Decision in One Sentence

Reduce active tuning to two repository-owned interfaces,
`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`, while retaining
historical/diagnostic wrappers and compatibility shims until MacroFinance and
dsge_hmc tests pass against the new contracts.

## Research/Engineering Intent Ledger

| Item | Contract |
|---|---|
| Main question | Can BayesFilter provide one coherent ordinary-HMC tuner and one transport-specific tuner without breaking MacroFinance or dsge_hmc? |
| Candidate mechanism | Extract shared geometry, mass, epsilon/trajectory, evidence, artifact, and replay mechanics from `hmc_kernel_tuning.py`; use the robust broad-grid orchestration as the ordinary-HMC candidate. |
| Expected failure mode | Consumer breakage from private imports, changed artifact payloads, target-scope drift, XLA/GPU policy mismatch, or a diagnostic route accidentally becoming active. |
| Primary promotion criterion | All active consumers call one of the two interfaces; repository and cross-repo contract tests pass; artifacts bind exact target/adapter/config/source identities and fail closed on mismatch. |
| Promotion vetoes | Missed call site/definition, broken MacroFinance or dsge_hmc tests, stale or caller-stamped artifact accepted, NumPy/runtime backend violation, missing target-scope match, unledgered route, or nonfinite/invalid evidence. |
| Continuation vetoes | Only a scientific or implementation invalidity that makes the migration question unanswerable: corrupted source, irreconcilable artifact semantics, or a failing independent contract with no safe compatibility route. A candidate tuner failure alone triggers repair. |
| Repair trigger | Add/repair a shim, migrate one consumer family, or split one mechanic with a focused test while keeping scientific target, evidence criteria, hardware class, and campaign budget unchanged. |
| Explanatory diagnostics | Candidate acceptance, R-hat/ESS, runtime, repair count, L/epsilon edge hits, source size, and import counts. They do not independently prove convergence or superiority. |
| Nonclaims | This plan does not establish posterior correctness, convergence, sampler superiority, production readiness, or scientific validity. It does not authorize deletion of historical artifacts. |

## Scope and Boundaries

### Active interfaces

#### 1. Ordinary HMC tuner

`bayesfilter.inference.tune_hmc_kernel` remains the compatibility name and
becomes the only active ordinary-HMC entry point. Its request must bind:

- adapter signature and target scope;
- coordinate/transport identity and parameter dimension;
- geometry hint source and mass artifact signature;
- epsilon/trajectory candidate policy;
- acceptance/health/veto policy;
- calibration versus untouched verification seeds/data;
- execution backend, dtype, TF32, XLA, and chain mode;
- bounded compute/attempt/timeout policy; and
- artifact schema/version and source dependency closure.

The robust broad-grid implementation may be selected as its internal strategy,
but it must not silently change the public semantics. The existing long
`hmc_kernel_tuning.py` route is the initial compatibility implementation while
the extracted route is proven.

#### 2. Fixed-transport tuner

`bayesfilter.inference.tune_fixed_transport_hmc_kernel` remains the only active
transport-coordinate tuning entry point. It must bind the base adapter,
transport manifest/signature, transformed target scope, fixed-coordinate mass
policy, and the same artifact/evidence lineage fields. Transport-specific grid
and parity mechanics remain internal to this route.

### Historical/diagnostic interfaces

The following remain importable only during migration, with explicit
`diagnostic_only`/`historical` role metadata and no default/admission authority:

- `tune_hmc_kernel_robust_broad_grid`;
- `run_fixed_metric_grid_search`;
- `run_operational_broad_grid` and process-parallel variant;
- `run_fixed_mass_hmc_tuning_budget_ladder`;
- `run_generic_hmc_tuning_orchestration` and `orchestrate_generic_hmc_tuning`;
- fixed-trajectory v2 and fixed-kernel arm routes;
- uncertainty retuning and capacity nomination helpers; and
- all old stage/phase functions that are not reached by the two public APIs.

Historical does not mean delete now. It means no new claim-bearing caller may
select it, and its artifacts cannot be silently upgraded to canonical status.

## Skeptical Plan Audit Before Execution

This audit is completed before implementation work.

| Risk checked | Finding | Plan response |
|---|---|---|
| Wrong baseline | Treating the new robust broad grid as automatically correct would hide its `use_xla=False` default, fixed L grid, and weak 500-draw qualification. | Use it as a candidate internal strategy only; require scope/default review and untouched verification. |
| Proxy promotion | Acceptance, ESS, R-hat, and runtime can nominate/explain but cannot prove posterior correctness or superiority. | Keep hard veto, descriptive, and statistical evidence classes separate in artifacts and tests. |
| Missing stop conditions | Extraction could become an open-ended rewrite. | Each phase has bounded file/consumer scope and a stop condition; no deletion until cross-repo suites pass. |
| Unfair comparison | Comparing old and new tuners with different target scope, seeds, data partitions, or backend would be meaningless. | Freeze target, seeds, candidate policy, dtype/backend, and calibration/verification split per migration fixture. |
| Hidden defaults | Fixed L values, acceptance bands, burn-in/result counts, and `use_xla` are inherited/convenience choices. | Record provenance for every numeric default; promote only after target-specific tests/diagnostics. |
| Stale context/artifacts | v1/v2 historical payloads may lack route identity. | Version the canonical schema and reject missing/mismatched identity; preserve old readers as historical. |
| Environment mismatch | MacroFinance and dsge_hmc use different layouts and may import BayesFilter through `PYTHONPATH`. | Run each suite from its own repository with explicit BayesFilter path and pinned environment manifest. |
| Artifact answers wrong question | A passing local unit test could leave external private imports broken. | Require cross-repo contract tests and import scans before each migration gate. |
| Resource risk | Full suites and GPU/XLA runs may be expensive. | Start with AST/import/contract tests and CPU-hidden unit tests; reserve trusted GPU/XLA canaries for explicit target routes. |

Audit result: the plan passes for a staged refactor. It would fail if it began
by deleting modules, changing artifact schemas in place, or promoting robust
broad-grid metrics as convergence evidence.

## Target Architecture

```text
public API: tune_hmc_kernel             public API: tune_fixed_transport_hmc_kernel
             |                                      |
             +----------- shared tuning contract --+
                         |
          request/scope validation and artifact identity
                         |
      geometry -> bootstrap -> mass -> epsilon/L -> verification
                         |
                evidence/state/artifact writer
                         |
              TFP full-chain execution authority

historical/diagnostic wrappers ----------------------+
  fixed metric | operational grid | budget ladder | generic grid | old phases
  delegate into shared mechanics but cannot issue canonical admission artifacts
```

The first extraction should be structural, not algorithmic: move code without
changing numerical behavior, then change policy only after parity tests pass.

## Phased Implementation

### Phase 0: Freeze and classify

Create `bayesfilter/inference/tuning_contract.py` containing typed request,
scope, evidence-role, and artifact-schema identifiers. Add a route registry
with exactly two `active` records and explicit historical/diagnostic records.
Add a committed inventory-generation script with explicit inclusion,
exclusion, false-positive, and archival-snapshot rules. Record a Phase-0
baseline before changing sampler behavior:

- BayesFilter collection and focused collection under the named `tfgpu`
  interpreter, including verification of the three repaired exports;
- CPU-hidden tests that fail closed by design and must be deselected/marked;
- MacroFinance collection under its pinned environment, including missing
  optional dependencies and import-order failures; and
- dsge_hmc collection with `BAYESFILTER_ROOT` set and `tests/archive` either
  explicitly ignored or separately recorded if it segfaults.

Collection is not a run-level migration baseline. Phase 0 must also execute the
plan-named focused MacroFinance and dsge_hmc commands and preserve exact pass,
failure, and skip counts plus assertion fingerprints. The 2026-08-17 baseline
has 11 MacroFinance and 2 dsge_hmc failures; they must be classified in a
drift-adjudication ledger before they can be used at a later gate:

1. Eight CCMA failures compare historical
   `bayesfilter.hmc_acceptance_evidence.v3` fixtures with the live v5 parser.
   Preserve the live v5 fail-closed rule. Either regenerate v5 evidence or use
   an explicit historical, non-promoting reader; never silently upgrade v3.
2. Two MacroFinance L10d failures expect
   `ccma_phase4y_stage_budget_v1` after constructing the current default
   `HMCStagedTimeoutPolicy`, whose committed default identity is
   `bayesfilter_hmc_emergency_stage_caps_v2`. Decide from the intended consumer
   contract whether to update the stale expectation or explicitly instantiate
   the older named policy; do not rewrite the serialized bound policy identity.
3. A third L10d failure is a brittle public-redaction assertion that rejects
   every textual occurrence of `0.25`, including legitimate budget-policy
   constants. Replace it with structural assertions that mass/covariance arrays
   and private fields are absent. It is not a timeout-policy drift.
4. Two dsge_hmc Rotemberg failures expect an older selection rule and 49
   candidates, while the committed BayesFilter policy and its own tests require
   `eligible_trajectory_acceptance_in_band_then_rhat_convergence_then_ess` and
   63 candidates. Adjudicate the intended consumer contract before migration;
   do not weaken the repository policy merely to satisfy a stale count.

Each ledger entry must cite the inspected producer and consumer authorities,
classify the consumer as stale or the BayesFilter change as incompatible, name
the repair and responsible phase, and record the rerun result. Owner direction
is required only if those authorities do not resolve a material intended-
contract choice; it is not a blanket requirement for correcting a demonstrably
stale test.

The three known inference export regressions were repaired before the Phase-0
gates: `HMCStagedTimeoutPolicy`,
`prepare_fixed_transport_hmc_adaptive_joint_grid_policy`, and the companion
`prepare_fixed_transport_hmc_joint_grid_policy` export must resolve through
both `bayesfilter.inference` and `bayesfilter`. Phase 0 must verify both
package layers and record the check; it must not reclassify these completed
repairs as future implementation work.

Deliverables:

- exact AST inventory script under `tests` or `scripts`;
- active-route discovery guard;
- public API and artifact schema documentation;
- source/import scan for BayesFilter, MacroFinance, and dsge_hmc;
- focused run-level baseline manifest; and
- completed drift classifications for every pre-existing focused failure.

Gate: the guard fails on an unclassified active tuning route, duplicate active
interface, or a new direct consumer of a historical route. The baseline
manifest, export repairs, and CPU/GPU marker policy must be recorded before
Phase 1 begins; later gates are evaluated relative to that baseline, not by
requiring an impossible "all tests pass unchanged" result. Collection-only
evidence cannot satisfy this gate. An unresolved choice about which side owns a
contract is allowed to remain open during structural extraction, but it blocks
the affected Phase 4-6 migration gate.

### Phase 1: Add compatibility shims

Keep existing function names and signatures. Make historical wrappers delegate
to the canonical request/evidence/artifact types and mark their payload role.
Expose stable replacements for dsge_hmc private imports, for example:

- `bayesfilter.inference.hmc_artifacts.mass_artifact_signature`;
- `bayesfilter.inference.hmc_geometry.BootstrapFixedMassAdapter`;
- typed public replay/admission helpers.

For one release/migration window, old private names re-export these replacements
with deprecation comments and tests. Do not modify historical artifact files.

Gate: no new test failure appears relative to the recorded Phase-0 run-level
baseline; every pre-existing focused failure has a completed classification and
named downstream repair. The three repaired exports must resolve through both
package layers before this gate closes. Focused MacroFinance and dsge_hmc
import/contract tests must run under their named environments and required
repository variables. Merely preserving an unexplained failure count is not a
passing compatibility result.

### Phase 2: Extract shared mechanics from `hmc_kernel_tuning.py`

Move, one family at a time, with compatibility imports left in place:

1. `hmc_geometry.py`: geometry configs/results, mass construction, curvature.
2. `hmc_bootstrap.py`: bootstrap config/result, fixed-mass latent adapter.
3. `hmc_mass_adaptation.py`: windowed mass stage and covariance updates.
4. `hmc_kernel_stages.py`: fixed-mass epsilon, frozen trajectory, candidate
   repair, and verification stage records.
5. `hmc_artifacts.py`: public/private projection, hashes, replay/admission,
   atomic writes.
6. `hmc_budget_policy.py`: geometry-scaled budgets and timeout closeout.

The extraction list also includes `hmc_kernel_selection.py` as the candidate,
handoff, repair, and sanitized-selection evidence family. Review its three
currently duplicated `_mass_artifact_signature` definitions and select one
canonical implementation with consistency tests before moving callers. The
direct dependency closure also includes `hmc_budget_contract.py`,
`hmc_route_contract.py`, `batched_value_score.py`, `hmc_coordinates.py`,
`hmc_diagnostics.py`, `hmc_posterior_diagnostics.py`,
`neutra_shared_procedure.py`, `runtime/__init__.py`, `native_tfp_hmc.py`, and
`neutra_hmc.py`; these are extraction dependencies, not optional follow-up
cleanup.

Use one authoritative artifact-helper home. The proposed `hmc_artifacts.py`
must either replace `hmc_tuning_artifacts.py` with a compatibility facade or be
renamed; do not create a third parallel artifact authority.

The original module becomes a compatibility facade importing these definitions.
No numerical default changes are allowed in this phase.

Gate per family: AST/import check, unit tests for shape/finiteness/signature
invariants, old/new function parity on deterministic fixtures, and no changed
artifact hash for a frozen fixture unless the schema version changes. The mass
family additionally must pass the explicit mass-matrix contract below.

### Mass-matrix and metric contract

Mass tuning is a first-class part of both interfaces, not an incidental detail
of geometry. The implementation must state which object each field means:

- `Sigma_theta`: stored position-coordinate covariance/preconditioner in the
  canonical parameter coordinates;
- the affine factor `F` with `F @ F.T = Sigma_theta`, including its declared
  orientation;
- the latent-coordinate transform induced by `F`; and
- the momentum covariance/kinetic precision actually used by the TFP HMC
  kernel after transformation.

Do not call `Sigma_theta` the posterior covariance merely because it was
estimated during warmup. A mass/preconditioning estimate can be wrong while a
correct HMC kernel still targets the right density, and a numerically valid
mass artifact does not establish posterior correctness. The canonical artifact
must bind the coordinate signature, adapter signature, position role,
covariance source, regularization policy, eigenvalue/condition summaries,
factor orientation, factor reconstruction tolerance, transform signature, and
the downstream momentum-metric convention.

The ordinary-HMC route may use geometry hints, negative-Hessian precision,
parameter scales, and windowed warmup covariance updates. The fixed-transport
route must retain its explicit fixed-coordinate mass policy unless a separate
reviewed transport mass policy is introduced; do not silently apply ordinary
windowed adaptation in transformed coordinates. Every mass update invalidates
the old epsilon/trajectory context and must issue a fresh seed and typed
handoff before step-size or trajectory tuning continues.

The joint mass/epsilon/`L` procedure is staged and conditional, not a claim of
global joint optimization. After the mass is constructed or updated, epsilon
must be retuned for that mass before trajectory selection. The compatibility
monolith currently tunes one epsilon at fixed mass and then selects among `L`
candidates; the robust broad grid instead runs independent dual averaging for
each `L`. The canonical request and artifact must identify which fresh-epsilon
policy was used. Results from the two policies are not interchangeable, and a
mass change invalidates either policy's prior epsilon/`L` evidence.

`bayesfilter/inference/mass_matrix.py` currently implements admitted mass
construction and regularization with NumPy. Under the repository NumPy policy,
this is migration debt, not an approved artifact-boundary exception. The Phase-2
mass extraction must provide the canonical construction, regularization,
factorization, summary, and artifact inputs through TensorFlow/TFP and Python
standard-library types. The NumPy implementation may remain only as an
explicitly independent diagnostic/reference authority and must not be imported
by either canonical runtime route. This Phase-2 repair is scoped to the mass
execution path; Phase 3 must still audit the full canonical dependency closure
for other NumPy runtime debt before admission.

The mass test family must include:

1. **Exact construction tests.** For a known covariance and precision, verify
   covariance/precision inversion, symmetry, Cholesky/factor reconstruction,
   whitening round trips, eigenvalue and condition summaries, and deterministic
   mass-artifact signatures. Verify geometry hint precedence: negative Hessian,
   supplied covariance, then parameter scales, with the configured fallback
   behavior recorded rather than silently selected.
2. **Windowed-estimator tests.** Feed deterministic warmup draws with a known
   covariance into Welford/shrinkage updates and compare the resulting
   covariance, regularization report, update windows, reset events, and final
   artifact to an independently calculated reference. Test both adaptive and
   `fixed_identity` routes; the latter must produce no mass updates. Use a
   separate non-diagonal rotated Gaussian fixture so diagonal-only tests cannot
   pass with a wrong implementation.
3. **Fail-closed tests.** Nonfinite, asymmetric, singular/indefinite, wrong
   dimension, wrong adapter/coordinate signature, stale shrinkage target,
   corrupted factor, impossible condition cap, and nonpositive eigenvalue
   inputs must hard-veto or raise according to the declared policy. Any
   regularization must be explicit in the artifact; it must not silently turn
   a failed mass estimate into a canonical success.
4. **Frozen-metric replay tests.** Reload the repository-issued mass artifact
   in a fresh process, reconstruct the transform and momentum metric, and
   verify exact signatures, factor orientation, coordinate round trips, and
   epsilon-context invalidation after a mass change. Caller-supplied or stale
   mass signatures must fail closed.
5. **Mass-versus-target holdout.** In the analytic Gaussian oracle, compare an
   identity metric, the supplied exact covariance, the warmup-adapted metric,
   and a deliberately geometrically mismatched but valid SPD metric, such as
   the analytic precision mistakenly supplied in the position-covariance role.
   For every arm, use the same target, starts, root-seed schedule, candidate
   policy, and holdout budget, but derive arm-specific fresh seeds and retune
   epsilon and `L` after binding that mass. The primary adequacy criterion is
   target-preserving holdout validity (finite states, target-status health, no
   divergences, R-hat/ESS gates, and analytic moment agreement), not closeness
   of an empirical mass to the true covariance. Predeclared analytic
   mass-distance, whitening, energy, trajectory, acceptance, ESS, condition,
   and runtime diagnostics must distinguish and explain the mismatched arm.
   They do not have to force repair solely because the SPD mass differs from
   the analytic covariance: correct HMC may retain the target and epsilon/`L`
   retuning may compensate. Repair fires only under its predeclared health,
   validity, or efficiency condition. No arm may be promoted as better without
   uncertainty evidence, and a failed mass candidate is a repair trigger rather
   than evidence against HMC itself.

The mass contract is satisfied only when these tests cover the ordinary
windowed-adaptive path and the fixed-transport fixed-mass path. Existing
geometry, bootstrap, and windowed-mass tests remain useful, but they do not
close the mass-versus-target holdout gate by themselves.

### Phase 3: Canonicalize ordinary-HMC orchestration

Implement one internal `_run_canonical_hmc_tuning(request)` and have
`tune_hmc_kernel` call it. Use the robust broad-grid sequence only where its
scope and policy are explicitly requested. Convert the old robust function into
a thin diagnostic wrapper or a compatibility alias; it must not issue a second
canonical artifact schema.

Before target-specific review of its numeric controls, generalize
`RobustBroadGridConfig`: the current implementation hard-rejects every L grid
except `(3, 5, 9, 13, 18, 25)` and every qualification rung except 500. These
must become explicit reviewed policy fields with provenance, rather than frozen
constraints that prevent a target-specific tuning scope from being expressed.

Gaussian and domain-specific fixtures are sufficient to gate the structural
refactor, but they are not sufficient to promote the robust route's numeric
defaults. Promotion of an `L` grid, acceptance/repair band, qualification rung,
or related robust default requires at least one non-Gaussian curved or
varying-Hessian target fixture, such as a banana target, with a predeclared
evidence contract and untouched verification, or an explicit owner waiver that
records the missing stress evidence and resulting nonclaims. This is a
default-promotion gate, not a Phase 0-2 extraction blocker. A hierarchical
funnel remains an additional stress diagnostic unless separately promoted to a
required target class.

Required behavior:

- target-specific tuning scope and disjoint calibration/verification data;
- repository-issued artifact identity, never caller-stamped;
- TensorFlow/TFP runtime; no NumPy numerical path in admitted execution;
- XLA default consistent with the BayesFilter policy, with explicit reference
  exception recorded when disabled;
- fresh seeds for tuning, repair, and untouched verification;
- typed hard veto, continuation veto, repair trigger, and explanatory roles;
- no posterior samples retained by tuning;
- no ranking claim from one seed/short chains; and
- deterministic tie-breaking with uncertainty metadata.

Gate: BayesFilter canonical API tests, artifact replay tests, route-ledger
tests, a deterministic synthetic fixture comparing old and new mechanics, and
the posterior-oracle adequacy gate below.

### Phase 4: Canonicalize fixed-transport orchestration

Keep transport-specific controls and parity checks but route them through the
shared request/evidence/artifact contract. Add exact transformed-scope checks,
transport manifest hash checks, and the same historical-artifact rejection.
Do not force LEDH/transport controls into ordinary HMC vocabulary.

Gate: existing `fixed_transport_hmc_tuning_tf` tests, dsge_hmc Rotemberg
NeuTra tuning smoke/contract tests, XLA route checks, and transformed-target
parity fixtures. The affine-Gaussian posterior-oracle case below is required
for the fixed-transport interface; a transformed-coordinate screen alone is
not sufficient evidence that the selected kernel targets the right measure.
The relevant focused run-level drift entries must be adjudicated and repaired;
an unchanged 24-pass/2-fail baseline is not sufficient to close this gate.

### Posterior-oracle adequacy gate

The repository already has standard-normal fixtures in
`tests/test_hmc_kernel_tuning_fixed_mass_step.py`,
`tests/test_hmc_budget_ladder.py`, and
`tests/test_fixed_trajectory_hmc_tuning.py`. They are useful mechanics and
real-TFP smoke tests, but they do not currently make analytic posterior
agreement a pass/fail criterion for either canonical interface. Add a
dedicated `tests/test_hmc_tuning_posterior_oracle.py` using a TensorFlow/TFP
adapter for the exact two-dimensional target

`theta ~ N(mu, Sigma)`

with nonzero `mu` and a non-diagonal positive-definite `Sigma`. Keep the oracle
parameters in the test fixture and compute the expected mean and covariance
from that same exact specification; do not use samples from the tuner as the
reference. The test family must include:

1. **Value/gradient oracle.** Compare the adapter value and total score with
   the closed-form Gaussian log density and score at fixed points, including
   batched inputs and a finite-difference or TensorFlow-gradient check. A
   failure here invalidates the fixture rather than the tuner.
2. **Ordinary-HMC holdout.** Tune on calibration seeds/starts only, freeze the
   selected artifact, then run an untouched real-TFP holdout with independent
   seeds and multiple chains. Check finite states, target status, native
   divergences, rank-normalized split/folded R-hat, bulk/tail ESS, and the
   sample mean/covariance against the analytic `mu`/`Sigma`. Tuning screens may
   nominate a kernel; only the untouched holdout can test this target-agreement
   question.
3. **Fixed-transport holdout.** Repeat the same test with a nontrivial affine
   transport whose transformed target has an analytically known mean and
   covariance. Verify both transformed-coordinate and base-coordinate
   identities, including the Jacobian/score composition and artifact scope
   match. An identity transport is a control, not sufficient coverage.
4. **Negative controls.** Deliberately use a wrong mean, covariance, Jacobian,
   or target score and require the oracle test to fail. This guards against a
   test that merely checks internal consistency of the same incorrect code.

The pass/fail tolerances, chain count, retained draws, seeds, and MCSE method
must be recorded in the fixture's evidence contract with provenance. Use
analytic moments and MCSE/uncertainty-aware intervals for continuous moment
checks; do not promote a one-seed mean, covariance, ESS, R-hat, or acceptance
value as evidence of adequacy. A short deterministic run may remain a smoke
test, but it cannot close this gate. The fixture must record calibration versus
holdout separation, actual command/environment, selected artifact hash, and
the explicit nonclaim that a Gaussian oracle does not prove correctness for
MacroFinance, dsge_hmc, nonlinear targets, or production readiness.

This is an engineering/numerical validity gate for the tuner on a known target,
not a claim that the tuner is statistically superior or universally adequate.

### Phase 5: Migrate MacroFinance

Migrate claim-bearing callers in this order:

1. MIDAS robust broad-grid driver to `tune_hmc_kernel` with an explicit robust
   strategy request or a documented diagnostic wrapper.
2. Ordinary MIDAS/CCMA/one-country calls to canonical `tune_hmc_kernel`.
3. Fixed-metric and operational neighbor-guard scripts to diagnostic wrappers
   or the canonical candidate protocol, preserving their nonclaims.
4. Replace source/private imports of `hmc_kernel_tuning` with public modules.
5. Migrate the budget-ladder callers in
   `mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning_v2_phase4_step_trajectory.py`
   and its Phase-5T loop/tests.
6. Migrate `orchestrate_generic_hmc_tuning` callers in
   `cross_country_multi_asset_bayesfilter_owned_hmc_client.py` and
   `..._mass_preconditioner.py`, plus their tests.

Update MacroFinance source-level tests to assert the canonical route and keep
historical scripts explicitly diagnostic. Do not convert historical result
artifacts into active evidence.

Gate: focused MacroFinance tuning tests, all contract/source scans, then the
full MacroFinance pytest suite. All 11 focused baseline failures must have their
adjudicated repair applied and pass, unless an explicitly historical fixture is
removed from the active gate and retained under a non-promoting historical
reader. "No worse than 90 passed/11 failed" is not a migration pass.

### Phase 6: Migrate dsge_hmc

Migrate BGS stage-C grid to the canonical ordinary-HMC candidate protocol or a
documented diagnostic wrapper, retaining its explicit aggregate policy. Migrate
Rotemberg fixed-transport smoke to the second public interface. Replace private
mass/bootstrap imports with public replacements. Update contract tests to check
the new module paths while preserving no-local-sampler, TensorFlow/TFP, XLA,
and artifact identity requirements.

Gate: focused BGS grid and Rotemberg contract tests, then all configured dsge_hmc
pytest paths (`tests`, `tests/contracts`, `tests/regressions`,
`tests/integration`, `tests/golden`, `tests/numerics`, `tests/extended`) with
`BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter`, the named `tfgpu`
interpreter, and `tests/archive` explicitly ignored while its segfaulting
baseline is unresolved. The two focused policy/count failures must be resolved
according to the drift-adjudication ledger; preserving a 24-pass/2-fail result
does not close this gate.

### Phase 7: Quarantine and cleanup

After two green cross-repo runs, remove historical routes from default
`__all__` exports or move them under an explicit `bayesfilter.inference.legacy`
namespace. Keep import shims for one documented compatibility window. Add a
test that a new claim-bearing source cannot import a historical route.

Only then consider deleting dead compatibility code. Historical artifacts and
readers remain preserved.

## Test and Verification Matrix

Commands are examples to be executed from clean processes with the appropriate
environment. CPU-hidden tests are engineering checks, not GPU evidence.

### Environment and baseline

Use the repository's TensorFlow environment explicitly; bare base `python` may
not have TensorFlow. For this checkout the audited interpreter is `tfgpu`
(Python 3.13.13, TensorFlow 2.20.0, TFP 0.25.0). If the project selects a
different environment, record its versions in the manifest.

Before Phase 1, run `--collect-only` in all three repositories under these
environments and record collection errors by category: pre-existing export
defects, intentionally GPU-required tests under CPU hiding, missing optional
dependencies, import-order/check-out resolution failures, and archive or
segfaulting tests. Also run the plan-named focused MacroFinance and dsge_hmc
commands and record pass/fail/skip counts and assertion fingerprints. This is a
dirty-worktree baseline: collected-test and run counts may drift as files
change. Before every migration gate, regenerate and timestamp both manifests;
collection comparisons use error fingerprints/categories and explicitly
excluded paths, while run-level comparisons require every tuning-contract drift
to be explained and resolved by its ledger. "No worse than baseline" prevents
regression during extraction but cannot promote a migration with known active
contract failures.

Recorded baselines after the public export repair (2026-08-16, refreshed
2026-08-17):

| Repository/scope | Recorded result | Interpretation |
|---|---|---|
| BayesFilter, CPU-hidden, two known GPU-only tests ignored | 7,452 tests collected; 3 unrelated collection errors | The refreshed dirty-worktree count is +5 from 2026-08-16; the three error fingerprints remain Zhao-Cui lane-B T2 score and two Kalman QR Phase-6 guard-state tests. Error categories and exclusions, rather than the raw count, are gate-relevant. |
| MacroFinance, six focused tuning files | 101 collected; 90 passed / 11 failed when run CPU-hidden in `tfgpu` | The repaired export resolves, but collection concealed active failures: 8 historical-v3/live-v5 acceptance-evidence failures, 2 staged-timeout identity expectation failures, and 1 unrelated brittle redaction assertion. These are four technical adjudication entries when combined with dsge_hmc, not one uniform BayesFilter drift family. |
| MacroFinance, full suite | 4,252 tests collected; 38 collection errors | Baseline evidence, not a tuning-plan failure. Known causes are missing `pandas` in `tfgpu` and order-dependent BayesFilter checkout-resolution failures. Resolve each root cause or record an explicit owner waiver before full-suite promotion. |
| dsge_hmc, three focused contract files with `BAYESFILTER_ROOT` | 26 collected; 24 passed / 2 failed when run CPU-hidden in `tfgpu` | The environment/path contract works. Both failures assert the older fixed-transport selection rule and 49-candidate grid rather than the committed 63-candidate BayesFilter policy. |

The three previously reported public export defects are repaired and verified:
`tests/test_fixed_transport_hmc_grid_policy.py` passes 24 tests, and the
MacroFinance L10d file collects 20 tests. This is engineering compatibility
evidence only.

### BayesFilter fast checks

```bash
cd /home/ubuntu/python/BayesFilter
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m compileall -q bayesfilter tests
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_hmc_robust_broad_grid.py \
  tests/test_hmc_kernel_selection.py \
  tests/test_hmc_mixed_candidate_handoff_policy.py
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_hmc_fixed_metric_grid_search.py \
  tests/test_fixed_transport_hmc_tuning.py \
  tests/test_frozen_kernel_validation.py
```

The mass-specific contract suite must be run before the posterior-oracle
holdout:

```bash
cd /home/ubuntu/python/BayesFilter
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_hmc_mass_matrix.py \
  tests/test_hmc_windowed_mass_adaptation.py
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_hmc_kernel_tuning_geometry.py \
  tests/test_hmc_kernel_tuning_bootstrap.py \
  tests/test_hmc_kernel_tuning_windowed_mass.py \
  tests/test_hmc_budget_ladder.py -k 'mass or covariance or metric or gaussian'
```

This command is contract/mechanics evidence. It does not close the
mass-versus-target holdout requirement below.

After Phase 3/4 creates the posterior-oracle fixture, run it separately from
the fast contract suite so a target-agreement failure cannot be hidden by
plumbing tests:

```bash
cd /home/ubuntu/python/BayesFilter
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_hmc_tuning_posterior_oracle.py
```

Until that file exists, the plan must report the posterior-oracle gate as an
open migration gap. The current Gaussian tests remain useful smoke/mechanics
coverage but do not close the gate.

If a named test file does not exist in the current checkout, the runner must
record the missing path and use the closest existing module test; it must not
silently claim the test ran.

### BayesFilter full unit suite

```bash
cd /home/ubuntu/python/BayesFilter
conda run -n tfgpu env CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests \
  --ignore=tests/test_neural_force_hmc_gpu.py \
  --ignore=tests/test_neural_force_training_gpu.py
```

### MacroFinance focused then full suite

```bash
cd /home/ubuntu/python/MacroFinance
conda run -n tfgpu env PYTHONPATH=/home/ubuntu/python/BayesFilter:$PWD CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/test_daily_asset_midas_bayesfilter_owned_tuning_execution.py \
  tests/test_daily_asset_midas_bounded_tuning_repaired_stack.py \
  tests/test_daily_asset_midas_l10c_bayesfilter_tuning_repair.py \
  tests/test_daily_asset_midas_l10d_bayesfilter_bootstrap_geometry_repair.py \
  tests/test_run_ccma_broad_fixed_metric_l_epsilon_search.py \
  tests/test_two_currency_double_zlb_dz5_neutra_fixed_metric_grid.py
conda run -n tfgpu env PYTHONPATH=/home/ubuntu/python/BayesFilter:$PWD CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests
```

There is currently no dedicated `test_daily_asset_midas_robust_broad_grid_tuning.py`
in MacroFinance. Phase 5 must add one covering the robust driver import,
configuration, progress/artifact handoff, and canonical-interface migration;
until then, the two existing MIDAS tuning tests above are only partial coverage.
The run manifest must list this missing test as a migration gap, not claim it ran.

### dsge_hmc focused then configured full suite

```bash
cd /home/ubuntu/python/dsge_hmc
conda run -n tfgpu env PYTHONPATH=/home/ubuntu/python/BayesFilter:$PWD BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/contracts/test_bgs_bayesfilter_stage_c_grid_tuning.py \
  tests/contracts/test_rotemberg_fixed_neutra_bayesfilter_xla_relaunch.py \
  tests/contracts/test_rotemberg_fixed_neutra_xla_gate.py
conda run -n tfgpu env PYTHONPATH=/home/ubuntu/python/BayesFilter:$PWD BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  --ignore=tests/archive
```

The dsge_hmc `pyproject.toml` declares pytest paths under `tests`,
`tests/contracts`, `tests/regressions`, `tests/integration`, `tests/golden`,
`tests/numerics`, and `tests/extended`; the final command must report those
paths, the explicit `tests/archive` omission, and any environment/import
blockers. The archive omission is required in this environment because its
module-level import currently segfaults; it is not evidence that archive tests
pass.

### Trusted GPU/XLA canaries

Only after CPU-hidden contract tests pass, run the smallest target-specific GPU
canary with `TF_FORCE_GPU_ALLOW_GROWTH=true`, repository memory-growth helper,
TF32/XLA settings, and a structured manifest. GPU commands require trusted or
elevated execution under repository policy. A GPU canary failure is not a
license to alter the canonical contract or promote CPU evidence.

## Artifact and Manifest Requirements

Every serious migration fixture must record:

- BayesFilter, MacroFinance, and dsge_hmc git commits/status;
- command, environment/conda env, Python/TensorFlow/TFP versions;
- CPU/GPU visibility, TF32, XLA, and memory-growth verification;
- target/adapter/transport signatures and exact tuning scope;
- calibration/verification partitions, seeds, budgets, and wall time;
- old/new route names and source hashes;
- artifact paths and SHA-256 hashes; and
- decision table: primary criterion, veto status, uncertainty, next action,
  and nonclaims.

Outputs must use unique versioned directories and must not overwrite prior
evidence.

## Stop Conditions and Rollback

Stop the migration phase, preserve artifacts, and do not delete code if:

- any consumer imports an unclassified route;
- old/new deterministic fixture outputs disagree outside the declared schema
  or numerical tolerance;
- an artifact accepts missing/mismatched scope or source identity;
- a cross-repo contract test fails for a reason not isolated to the shim;
- a GPU process lacks verified memory growth before TensorFlow initialization;
- a test command cannot establish which paths ran; or
- the remaining choice changes scientific target, data, hardware class, privacy
  boundary, or campaign budget.

Repair by restoring the compatibility facade and rerunning the focused gate.
Do not use destructive git commands or overwrite historical artifacts.

## Post-Plan Red-Team Review

Strongest alternative explanation: the apparent chaos may be intentional
support for materially different target/coordinate domains, not merely
duplication. The plan therefore preserves a separate fixed-transport interface,
keeps LEDH and NeuTra outside this consolidation, and treats operational and
fixed-metric routes as diagnostic until consumer evidence proves otherwise.

Result that would overturn this plan: a consumer requires a third genuinely
different numerical contract with independent target identity and artifact
semantics, or extracted ordinary-HMC mechanics cannot reproduce the existing
public route under frozen fixtures. In that case add a reviewed interface or
stop extraction; do not hide the difference behind a generic callback.

Weakest evidence: this is a source/consumer audit with focused run-level
consumer tests, not green full repository suites, and numeric defaults in the
robust route have not been target-specifically tuned. The first implementation
phase must therefore be contract and parity tests, not a scientific promotion
run.

## Acceptance Checklist

- [ ] Exact AST inventory regenerated and no definition/call-site gap found.
- [ ] Two active routes and all historical routes classified by the route guard.
- [ ] Compatibility shims cover MacroFinance and dsge_hmc private imports.
- [ ] Focused MacroFinance and dsge_hmc run-level baselines are timestamped,
      and every pre-existing failure has a source-anchored adjudication.
- [ ] Extracted mechanics reproduce frozen deterministic fixtures.
- [ ] Canonical mass construction is TensorFlow/TFP; NumPy remains independent
      diagnostic/reference code only.
- [ ] Mass construction, windowed estimation, regularization, signatures,
  replay, invalidation, and fail-closed cases pass.
- [ ] Mass-versus-target holdout compares identity, exact, adapted, and valid
  mismatched-SPD metrics with fresh epsilon/`L` retuning, under matched
  root-seed schedules/budgets and without descriptive-only ranking.
- [ ] Ordinary-HMC artifact scope/source/transport identities fail closed.
- [ ] Fixed-transport artifact and XLA/parity checks pass.
- [ ] Analytic Gaussian value/gradient, ordinary-HMC holdout, affine-
  transport holdout, and negative-control oracle tests pass with recorded
  uncertainty and calibration/holdout separation.
- [ ] Any promoted robust numeric default has non-Gaussian curved/varying-
  Hessian stress evidence or an explicit owner waiver and nonclaims.
- [ ] MacroFinance focused and full suites pass.
- [ ] dsge_hmc focused and configured full suites pass.
- [ ] New claim-bearing sources cannot import historical routes.
- [ ] No default/readiness/scientific claim is made from tuning-only evidence.
