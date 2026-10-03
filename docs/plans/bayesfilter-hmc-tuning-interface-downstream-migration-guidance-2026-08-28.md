# BayesFilter HMC Tuning Interface Downstream Migration Guidance

Date: 2026-08-28

Status: `READY_FOR_DOWNSTREAM_MIGRATION`

This dated note records what MacroFinance and dsge_hmc need to change after the
BayesFilter implementation is committed. It does not edit either downstream
repository, select a backend lock, or certify an HMC result.

## Version Snapshot

| Repository | Inspected commit | Relevant state |
| --- | --- | --- |
| BayesFilter | baseline `1ef8876666ea05698b3fa4e46a1d6c10a721fad7`; interface implementation `0da727c49d898478b3259189b846a029280849d8`; BGS mechanics-handoff correction `2315e2b28d5c6b6520df84681b6cb358f9808e36` | capability registry v1; runner binding v2; TensorFlow tuning schema v2; two active public tuners |
| MacroFinance | `98cca3bbed5289c770ff67d174808357ac8fd595` | ordinary tuner consumers; no direct neural-force runner in the scoped source search; dirty worktree preserved |
| dsge_hmc | `da060c6b4952925b7a1c58bb969300aea56e45c8` | integration plan uses the typed binding, but executable BGS stage still imports the low-level runner; dirty worktree preserved |

The inspected uncommitted dsge_hmc lock names BayesFilter
`4c2b2d7856b9f2cb6882e8f0b32592da81720840`. This work neither selected nor
edited that value.

## Compatibility Contract

A downstream consumer must bind all of the following, not just an exported
function name:

1. the exact BayesFilter Git commit;
2. capability registry schema
   `bayesfilter.hmc_tuning_capability_registry.v1`;
3. runner-binding schema `bayesfilter.hmc_tuning_runner_binding.v2` when a
   typed field binding is used; and
4. the short-name capability record returned by
   `hmc_tuning_interface_capability(interface_name)`.

The only active artifact-authority routes are the public dispatcher
`bayesfilter.inference.hmc_tuning_dispatch.tune_hmc_kernel` and
`bayesfilter.inference.fixed_transport_hmc_tuning_tf.tune_fixed_transport_hmc_kernel`.
The legacy ordinary symbol remains a compatibility delegate. A
`public_tuner` classification alone is insufficient: the route must also be
active, `tested_supported`, and `artifact_authority=True`.

The TensorFlow configuration exposed through the ordinary dispatcher has
`diagnostic_only` and `candidate` roles. Only a candidate that passes the
declared metric-update, final-divergence, and acceptance screen can build a
retained runner, and that runner preserves the exact binding and affine
geometry. This is frozen-mechanics authority only. The artifact explicitly has
no posterior-admission authority; retained R-hat/ESS and scientific
interpretation remain separate. Non-XLA execution does not require XLA
qualification.

Stable downstream policy text should remain short:

```text
Before changing an HMC consumer, inspect the capability registry at the pinned
BayesFilter commit. Use only an active, tested artifact-authority tuner matching
the target and coordinates. A chain runner or stage helper is not a tuner. Fail
closed when no supported route matches, and never route an arbitrary force
through the frozen-transport tuner.
```

Link detailed mechanics to `docs/reference/hmc-tuning-interface.md`; do not
copy the stage description into downstream policy.

## Search Evidence

The scoped searches were read-only and were run against the dirty downstream
worktrees without changing them.

MacroFinance direct-runner search:

```bash
cd /home/ubuntu/python/MacroFinance
rg -n --glob '*.py' --glob 'AGENTS.md' \
  'run_full_chain_neural_force_hmc' .
```

Result: no matches. The current Phase 14 ordinary call sites are:

- `daily_asset_midas_identifiable_multi_asset_expansion_phase14_hmc.py:225`
  and `:1036`;
- `scripts/run_daily_asset_midas_phase14_covariance_first_tuning.py:2072`
  and `:2261`; and
- `tests/test_daily_asset_midas_phase14_covariance_first_tuning.py:248`.

They were obtained with:

```bash
rg -n 'run_full_chain_neural_force_hmc|tune_hmc_kernel|bind_neural_force_hmc_tuning_runner' \
  daily_asset_midas_identifiable_multi_asset_expansion_phase14_hmc.py \
  scripts/run_daily_asset_midas_phase14_covariance_first_tuning.py \
  tests/test_daily_asset_midas_identifiable_multi_asset_expansion_phase14_hmc.py \
  tests/test_daily_asset_midas_phase14_covariance_first_tuning.py
```

The corresponding dsge_hmc command was:

```bash
cd /home/ubuntu/python/dsge_hmc
rg -n 'run_full_chain_neural_force_hmc|tune_hmc_kernel|bind_neural_force_hmc_tuning_runner' \
  scripts/run_bgs_full_estimation_hmc_stage.py \
  docs/plans/bgs-bayesfilter-adaptive-hmc-interface-integration-plan-2026-08-28.md \
  tests/contracts/test_bgs_full_estimation_master.py
```

The executable BGS stage imports the low-level runner at line 132 and registers
it at line 237. The current integration plan names the typed binding and public
tuner throughout; the master contract test still expects the low-level symbol
at line 81. Generated experiment artifacts and historical notes were excluded
from these consumer-code classifications.

## MacroFinance Disposition

The MacroFinance feedback is correct on all three material points:

1. The guide's former statement that the tuner simply "owns geometry" hid the
   caller's responsibility for the center and optional Hessian, covariance, or
   parameter-scale hypothesis. The revised guide now gives precedence,
   coordinate, validation, fallback, and covariance-first requirements.
2. Nonidentity durable replay was defective at the BayesFilter boundary. The
   repaired result path consumes the validated geometry already stored in the
   in-memory result and preserves initial and adapted mass signatures through a
   JSON round trip without running tuning or HMC.
3. The current Phase 14 failure is separate. Its result note says the local
   covariance was usable, but the MacroFinance wrapper widened incumbent
   eligibility beyond BayesFilter's declared candidate set. The provenance
   gate fired before `tune_hmc_kernel`: tuner call count was zero and no HMC
   transition or kernel existed.

Required MacroFinance work:

1. Repair the Phase 14 incumbent-eligibility contract prospectively. Do not
   attribute that wrapper defect to HMC tuning or weaken the predeclared gate
   retroactively.
2. Keep the accepted center and covariance in the same unconstrained adapter
   coordinates and pass them together as `initial_position` and
   `initial_covariance`.
3. Add a pinned capability test that checks the two active authority routes,
   the qualified dispatcher path, schemas, geometry ownership, fresh
   verification, disabled ordinary tuning ESS admission, and the rule that
   acceptance alone cannot hand off.
4. Branch on the typed final status and private admitted mechanics, then persist
   durable mechanics before process exit. Do not treat redacted public JSON as
   replayable mass state.
5. Run focused wrapper and replay tests against the selected BayesFilter commit.
   No HMC campaign is needed for this compatibility gate.

The relevant failed-run diagnosis is
`docs/plans/daily_asset_midas_identifiable_multi_asset_expansion_phase_14_covariance_first_tuning_campaign_result_2026_08_28.md`
inside MacroFinance.

## dsge_hmc Disposition

The current downstream integration plan has already corrected its route design:
it constructs `FrozenPositionOnlyForce`, `FrozenTargetPotential`, and a
repository-issued binding, then calls the public `tune_hmc_kernel`. Our earlier
draft statement that the plan rejected a binding was stale and is withdrawn.
The executable BGS stage has not yet made that migration.

Route selection is conditional on the field's actual meaning:

- If the BGS quantity is the exact score of the endpoint target, use the
  default ordinary runner and do not add a neural-force binding.
- If it is a different frozen deterministic position-only proposal field, label
  it with
  `DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS`, bind the exact
  endpoint potential, and pass the repository-issued binding to
  `tune_hmc_kernel`.
- A genuine frozen nonlinear transport with its Jacobian-adjusted transformed
  target belongs to `tune_fixed_transport_hmc_kernel`. An arbitrary field does
  not.

For the deterministic-field branch, the exact signs are
`potential = -log_target` and `proposal_force = -reported_log_target_field`
under the kernel's `p <- p - epsilon * force` convention. The field and endpoint
potential must share raw coordinates. Their identities, semantic label, chart
Jacobian declaration, affine constant-Jacobian convention, target scope, and
source closure are part of binding v2.

Required dsge_hmc work:

1. Select the exact BayesFilter correction commit in the BGS backend lock only
   after the user explicitly chooses that revision, then refresh preflight.
2. Keep claim-bearing BGS execution on the public dispatcher and typed binding;
   the historical direct-runner worker must remain unreachable from the master.
3. Retain negative tests for a bare callback, direct identity-mass fallback,
   missing/mismatched binding identity, wrong field semantics, coordinate or
   chart-Jacobian mismatch, and any retained handoff after failed required
   verification.
4. Retain positive tests that the same binding reaches tuning, artifact reload,
   retained pilot, and continuation, and that the binding hash is serialized.
   Those tests establish route mechanics, not posterior convergence.
5. Treat the completed one-draw BGS canary as historical diagnostic evidence.
   It failed the acceptance and covariance-rank screens and cannot authorize
   retained sampling under schema v2.

The TensorFlow-only BGS interface blocker is closed by the corrected public
route. This does not close target-specific Dynare parity, exact backend-lock
selection, campaign budgeting, or retained posterior assessment. A copied
DSGE-local tuner or a silent NumPy exception remains forbidden.

Do not update `config/bgs-backend-lock.json` until a later BayesFilter path
actually satisfies the downstream backend policy, its focused tests pass at an
exact pushed commit, and the user selects that revision. This note authorizes
no retained HMC run or posterior claim.

## Status Table

| Consumer | Correct route | Current status | Remaining work |
| --- | --- | --- | --- |
| MacroFinance ordinary targets | `tune_hmc_kernel` | route choice correct; Phase 14 wrapper failed before tuner | repair eligibility contract, pin schema/commit, test covariance and durable replay |
| dsge_hmc BGS exact score, if established | default `tune_hmc_kernel` | conditional route only | prove the exact-score classification and satisfy backend admission |
| dsge_hmc BGS non-gradient field | binding v2 passed to `tune_hmc_kernel` | current worker/master migrated and focused contracts pass; correction commit not selected in lock | select exact commit, refresh preflight, close Dynare prerequisite, then budget candidate tuning |
| Frozen nonlinear transport consumers | `tune_fixed_transport_hmc_kernel` | separate route | verify transport-specific target, policy, schema, and pin |

BayesFilter tests establish only BayesFilter interface behavior. Each consumer
must run its own contract tests against its selected commit.
