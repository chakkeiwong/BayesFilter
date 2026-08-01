# Phase 5 Result: Typed Baseline Adoption And Adversarial Validation

Date: 2026-07-11

Status: `PASSED_TO_PHASE6_AUTHORITY_IMPLEMENTATION_NO_RUNTIME`

## Direct Verdict

The human-approved typed-identity baseline migration passed its local
engineering, integrity, and governance gates. The refreshed replay is now the
active V2 validation baseline. Live reconstruction matches its adopted
transition, serious-execution, smoke-execution, provenance, canonical-payload,
and exact-file identities.

This result does not establish equality with the unavailable historical private
transition. That comparison remains `unsupported`. It also does not authorize
or report an HMC transition, worker, smoke, burn-in, retained sample, Phase 8,
or NeuTra run. The V2 config and adoption record both carry
`runtime_authority=false`, and the controller refuses runtime before creating
outputs or workers.

## Approval Consumed

The recorded human statement is exactly:

`I approve PROPOSE_TYPED_IDENTITY_BASELINE_MIGRATION bound to certificate sha256:684c5ae23c48f0d233fb8797927cb1574836ba0c0af3bf362c71b55e1aa1fc7f.`

That authority was consumed only for the bounded Phase 5 action reviewed in
the Phase 4 result: create and validate a new V2 baseline while preserving the
historical V1 evidence. It was not treated as smoke or serious-runtime
authority.

## Implemented Scope

- Added strict V2 baseline, adoption-record, preflight-report, artifact-
  reference, and terminal-manifest schemas in
  `bayesfilter/inference/hmc_identity_adoption.py`.
- Added one shared side-effect-free live replay/identity builder used by Phase
  3 evidence, V2 preflight, and worker initialization in
  `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py`.
- Made the typed V2 config the validation default while retaining the immutable
  historical V1 config as `HISTORICAL_V1_CONFIG_PATH`.
- Preserved the V1 validator as a named compatibility lane that still fails
  exactly with `public final kernel hash mismatch`.
- Required V2 preflight to reconstruct transition, serious/smoke execution,
  provenance, and canonical/exact-file integrity from live governed inputs.
- Classified three legacy whole-payload differences as
  `historical_audit_only`; they remain visible and are not suppressed.
- Added strict mutation, ownership, tamper, source-copy, redaction,
  compatibility, and no-runtime tests.
- Made `run_phase7` reject the active V2 config before output creation or
  worker creation because `runtime_authority=false`.

## Persistent Evidence

| Artifact | Embedded artifact hash | Exact file SHA-256 | Bytes |
| --- | --- | --- | ---: |
| V2 config: `docs/benchmarks/configs/multidim_lgssm_phase7_typed_identity_baseline_2026_07_11.json` | `sha256:bd127c2eb4e554c241a9f38111b5d832cb8ae9132429332abec724b9b2d39a6a` | `9270ec429a4b49e19f5ac6492e146bb1010e07c4ea0aa17600294e6c41db7ca8` | 12560 |
| Adoption record: `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/typed_identity_baseline_adoption_record.json` | `sha256:fc144b4af9f4963fdc3b96bc94065b245e7fe312c89936bd9ab861bc348b596e` | `73a41f8eadc98554a0479a3f7c79d0c81023d02f72649d4ef5d4d9de83a9bf01` | 4920 |
| Preflight: `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/typed_identity_baseline_preflight.json` | `sha256:e8e13bd7c7fc635424dd4401cf835dae6367f1be37386f3df45caaa3ef4a497e` | `373737f7db068b469050dd071718c5d2e2215919537cc5cdc50e620cd0a4729e` | 3918 |
| Terminal Phase 5 manifest: `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase5_output_integrity_manifest.json` | `sha256:32e97d69595029423fbe4e22f714c3596c8a4c3d9b3aabd6b2c600d279355bc0` | `41426951d25d02a7efbbd595e6edb0a81039fb297ce3ebd05e1695207fda4871` | 1196 |

The Phase 5 evidence graph is acyclic. The V2 config binds only pre-existing
Phase 4 evidence. The adoption record binds the exact V2 bytes. The preflight
binds the V2 config and adoption record. The terminal manifest binds those
three completed outputs. No completed Phase 5 artifact will be mutated to add
runtime authority.

The immutable V1 config exact file SHA-256 remains
`746b001d3facb771b3b57b032a212683743187deb944e6fae8eb577af073c0b8`.
All Phase 3 and Phase 4 protected artifact hashes remained unchanged.

## Identity Result

| Identity | Adopted and live-reconstructed value | Status |
| --- | --- | --- |
| Transition | `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a` | `equal` for the refreshed V2 baseline |
| Serious execution | `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4` | `equal` for the refreshed V2 baseline |
| Smoke execution | `sha256:fc85f9b1e0bb406593de9f5b8195ced6e86b10ee8fd549b1ecd1a8a24d6ac604` | `equal` for the refreshed V2 baseline |
| Selection provenance | `sha256:8d154c31b403ae2415a83f05e4848dc9a37df59b8fa51d5c12d88f6bfe46d32f` | `equal` for the refreshed V2 baseline |
| Complete tuning payload | `sha256:735e0178a76861382d49521aa726dcb3f73a8b0bd15a130775b6d74d978fc70d` | `equal` for the refreshed V2 baseline |
| Replay canonical payload | `sha256:14f252578e6d8e6f09194a0e1bef6af04f24fa18183dfae3620628c74f473c45` | `equal` for the refreshed V2 baseline |
| Replay exact file | `40c6a8d2cea8c55b5e923dace2bc3500a274b29a4abdd3ceed5b83c773f9ac0c` | `equal` for the refreshed V2 baseline |
| Historical vs refreshed typed transition | unavailable historical private transition | `unsupported` |
| Historical vs refreshed typed execution | no historical typed execution contract | `unsupported` |

The three historical whole-payload values remain `different` and
`historical_audit_only`: public final kernel, private-loop final kernel, and
selected trajectory. They are evidence of the old schema/provenance history,
not a typed mechanical veto after the approved new-baseline adoption.

## Checks Actually Run

All checks were deliberate CPU-only engineering checks with
`CUDA_VISIBLE_DEVICES=-1`, `MPLCONFIGDIR` under `/tmp`, and no sampler runtime.

| Check | Result |
| --- | --- |
| Python compilation for Phase 2-5/controller modules and tests | Passed |
| Focused Phase 5 suite | `26 passed` |
| Combined Phase 2-5/controller gate, final closeout run | `140 passed, 2 warnings in 29.92s` |
| Existing TFP warnings | Two `distutils.version` deprecation warnings only |
| Phase 3 governed-input and terminal-output verification | Passed |
| Phase 4 source/certificate/terminal-output verification | Passed |
| Phase 5 config/adoption/preflight/terminal-manifest verification | Passed |
| V2 strict unknown-field, type, reviewed-value, and tamper tests | Passed |
| V1 exact compatibility failure | Passed: `public final kernel hash mismatch` |
| V2 runtime refusal before output/worker creation | Passed |
| Public redaction and forbidden-mechanics scan | Passed |
| HMC transition, worker, smoke, burn-in, retained sampling | Not run |

`ruff` is unavailable, so no `ruff` result is claimed. TensorFlow emitted
CPU-hidden CUDA factory-registration messages during import; they are not GPU
health or GPU execution evidence.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Worktree | Dirty; in-scope migration files plus unrelated user-owned LEDH/QR changes preserved |
| Command class | CPU-hidden compilation, pytest, live replay reconstruction, JSON/schema/integrity validation, and hashing only |
| Python | `3.11.14` |
| TensorFlow / TFP | `2.19.1` / `0.25.0` |
| CPU/GPU status | CPU only by explicit `CUDA_VISIBLE_DEVICES=-1`; no GPU evidence |
| XLA/JIT | Configuration and typed identity checked; no XLA HMC transition executed |
| Data | Exact governed deterministic LGSSM fixture and replay protected by Phase 3/5 manifests |
| Random seeds | Configured seeds serialized and checked; no stochastic transition ran |
| Wall time | Final combined pytest `29.92 s`; no research runtime |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase5-adversarial-validation-subplan-2026-07-11.md` |
| Result | This file |

Final scoped source/test identities before independent review:

| File | SHA-256 |
| --- | --- |
| `bayesfilter/inference/hmc_identity_adoption.py` | `109013d5feeefc28fd3d0e5423d6e25c92830c7820bc5d765304ac2f1a89b890` |
| `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py` | `4988e83ba57a96a15ba60cb58bdf11c2b589d4033369233d6a05e21b2d7e0431` |
| `tests/test_hmc_identity_adoption.py` | `cc37c9c7430d1d9f8fb90f6ccdd4f98c0b1a4e8fd9dc81445e71fd76826ad274` |
| `tests/test_hmc_identity_integration.py` | `753aba46142edc61d189762843ac283bf881c93c0fad06b6c81f1139f0921a21` |
| `tests/test_deterministic_lgssm_hmc_phase7_tf.py` | `24fdd6c9d3852ad136a7f7baf286e734a2deda09bd1ee7dfeebb7f2a8d2b187c` |

## Decision Table

| Decision | Status |
| --- | --- |
| Phase 5 primary criterion | Passed: approved V2 live typed identities and exact artifacts agree; ownership/tamper/no-runtime tests pass. |
| Engineering correctness | Passed for local baseline materialization and preflight. |
| Historical typed equality | `unsupported`; not claimed. |
| Runtime authority | Absent. V2 config and adoption record are terminal with `runtime_authority=false`. |
| Smoke readiness | Phase 6 authority/launcher implementation and review are still required before an exact approval request can be formed; no smoke result exists. |
| Serious Phase 7 readiness | Not established and not authorized. |
| Next justified action | Review this result and Phase 6 subplan, implement/review the fail-closed smoke-only authority and launcher without runtime, then request exact human approval for only the frozen two-worker CPU/XLA smoke. |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard-veto screen | No Phase 5 engineering or integrity veto fired. HMC runtime vetoes were not evaluated because no transition ran. |
| Viable candidate | The refreshed typed baseline is admitted to a separately authorized mechanics smoke, not to serious sampling. |
| Statistically supported ranking | Not applicable; no sampler comparison or stochastic result exists. |
| Descriptive-only differences | Legacy whole-payload differences and prior acceptance/timing remain explanatory or historical only. |
| Default-readiness | Not evaluated and unsupported. |
| Next evidence needed | A separately approved tiny actual-target two-worker CPU/XLA smoke with dedicated artifacts and a terminal result review. |

## Post-Run Red Team

Strongest alternative explanation: the refreshed baseline may be internally
consistent while the unavailable historical private transition was materially
different. That possibility remains explicit as `unsupported`; this is a new
approved baseline, not a historical-equivalence finding.

What would overturn this result: drift in a governed input or terminal Phase 5
artifact, a live typed mismatch, an undisclosed transition-bearing field, a
public protected-value leak, or evidence that V2 can create an output or worker
without separate authority.

Weakest evidence: the historical comparison cannot be repaired without a
genuine historical private transition artifact.

## Next-Phase Handoff

The Phase 6 subplan is
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-subplan-2026-07-11.md`.
Preflight is already complete and must not be relabeled as a smoke. Phase 6
must first implement and locally test a separate smoke-only authority record
and dedicated-output launcher, freeze the exact command and proposal manifest,
and pass independent implementation/proposal review. Only then must the
supervisor stop and ask the human to approve the exact smoke decision bound to
that terminal proposal manifest. That approval cannot authorize serious
sampling.

## Nonclaims

- No historical/refreshed typed transition or execution equality is proved.
- No HMC transition, worker, smoke, burn-in, retained sample, Phase 8, or
  NeuTra execution occurred.
- No convergence, posterior recovery, ranking, production, default, GPU,
  package, or scientific claim is made.
