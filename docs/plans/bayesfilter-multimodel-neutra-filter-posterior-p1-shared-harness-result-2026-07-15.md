# P1 Result: Shared Multi-Model NeuTra Campaign Harness

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `P1_COMPLETE_REVIEWED_REOPENED_STATUS_IDENTITY_REPAIR_CLOSED`

## Decision

P1 admits the shared campaign harness for later target-specific phases. The
harness now issues a repository-owned typed target identity only after a
complete mathematical SSM contract passes independent posterior recomposition.
It binds the mathematical target, dtype, adapter signature, inspected batched
training callable and dependency closure, inspected HMC value/score callable
and dependency closure, scope, and recomposition admission.

All eleven declared nonlinear model/filter cells remain `TARGET_BLOCKED`. P1
issued no model-cell target signature and ran no model-cell HMC or training. Its
only positive runtime was the complete synthetic Gaussian canary.

## Primary Evidence

Terminal output root after the common status-identity reopen:
`docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p1/attempt-06-20260715T101223Z/`

| Check | Result |
| --- | --- |
| P0 registry replay | 11 unique cells; all `TARGET_BLOCKED`; zero model signatures |
| Independent posterior recomposition | Passed with separately inspectable prior, likelihood, and nonzero chart-Jacobian terms |
| Typed identity mutation guards | Prior, data, model, filter, chart, dtype, adapter, batch callable, HMC callable, hard-veto status callable, and cross-target artifact negatives pass |
| Circular recomposition guards | Direct or indirect reuse of the production final assembler is rejected |
| Batching policy | Graph-native rank-2 target, XLA, no scalar fallback, no callback, no mapped scalar target |
| Training policy | One 64-step batched GPU/XLA `tf.while_loop` invocation; verified memory-growth handoff; fresh output root |
| Frozen transport | Loaded only under target signature `888346dda772d566a2bb4417769f3185a1154c5a250d5f4a487e752fdd357fbb` |
| Transformed HMC health smoke | 16 draws, four chains, finite, all moved, no energy-error divergence, all status valid |
| Warm-up/retained policy | Shared controller tests and typed sequential integration preserve disjoint archives and exclude warm-up from posterior draws |
| CPU sample generation | Stateless fixed batches are worker-count invariant and TensorFlow-batched |
| Cell-state policy | P0-blocked cells cannot advance or reject recipes; recipe outcomes remain separate from cell rejection; events can persist append-only |
| Focused/compatibility tests | 72 passed under `CUDA_VISIBLE_DEVICES=-1`; two TFP deprecation warnings only |
| Artifact integrity | 11 recursively emitted files match `artifact_hashes.json` |
| Static policy audit | No NumPy import, TensorFlow host callback, mapped scalar target, or Python loop in the audited active tensor functions |

## Trusted GPU Manifest

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER, compute capability 8.9 |
| TensorFlow / TFP | 2.19.1 / 0.24.0 |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| XLA | Requested and runtime log recorded `Compiled cluster using XLA!` |
| Dtype | `float64` |
| TF32 flag | Enabled and recorded; no claim that every FP64 operation used TF32 |
| Memory growth | Verified before logical-device initialization |
| Allocator peak | 35,840 bytes for this tiny canary; explanatory only |
| Batch target first compile/run | 0.1362 seconds |
| Batch target warm run | 0.0039 seconds |
| Training compile/run | 2.1340 seconds |
| Transformed HMC compile/run | 2.3097 seconds |
| Total canary wall time | 7.7753 seconds |

Training loss moved from 1.6369 to 1.0756 and the health-smoke acceptance was
0.984375. Both are explanatory only. The canary is too short and too simple to
support training-quality, convergence, superiority, or nonlinear-model claims.

## Attempts And Repairs

| Attempt | Classification | Result | Repair |
| --- | --- | --- | --- |
| 01 | `INFRASTRUCTURE_LAUNCH_PATH` | Script-path invocation could not import repository package; no workload ran | Resolve repository root before BayesFilter imports |
| 02 | `HARNESS_INTEGRATION_STATUS_TELEMETRY` | GPU/XLA training completed; transformed HMC stopped before sampling for missing base status telemetry | Add graph-native synthetic status telemetry; retain hard status gate |
| 03 | `IDENTITY_INTEGRATION_TERMINAL_AUDIT` | Runtime passed, but terminal audit found HMC callable absent from typed digest | Bind and replay both training and HMC callable surfaces |
| 04 | `TERMINAL_CONTRACT_COVERAGE_AUDIT` | Runtime passed; review found memory-growth handoff, append-only state event, recursive hash, and typed archive-integration coverage should be explicit | Enforce/cover these boundaries without changing target or method |
| 05 | `TERMINAL_PASS` | Complete strengthened contract passed | None |
| 06 | `COMMON_STATUS_IDENTITY_REOPEN_PASS` | P2 adapter design exposed that hard-veto status telemetry was not yet identity-bound; three-surface identity and fresh canary pass | Bind and mutation-test `target_status_telemetry` |

The detailed record is
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p1-repair-record-2026-07-15.md`.
All attempts remain preserved. Total trusted GPU work was seconds, far below the
two-hour P1 bucket.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Dirty worktree | Shared dirty worktree with concurrent lanes; scoped paths only |
| CPU checks | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-mpl pytest -q tests/test_neutra_campaign.py tests/test_multimodel_neutra_p1_canary.py tests/test_neutra_batching.py tests/test_neutra_training.py tests/test_neutra_hmc.py tests/test_tensorflow_gpu_memory_policy.py tests/test_dense_iaf_neutra_artifact_loader.py tests/test_fixed_transport_hmc_binding.py` |
| GPU command | Exact command in terminal `run_manifest.json` |
| Seeds | Training `(20260715, 501)`; transformed HMC `(20260715, 502)` |
| Data | Synthetic canary only: `p1-synthetic-gaussian-observation-v1` |
| Plan | `docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p1-shared-harness-subplan-2026-07-15.md` |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit P1 shared harness and enter P2 target repair | Passed | No shared-harness veto | No declared nonlinear posterior target has yet been frozen or run | Repair/freeze `SVX-SGQF` and `SVX-ZC` targets before any HMC/training | No nonlinear cell, training recipe, convergence, filter accuracy, or scientific claim |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Shared identity, recomposition, state, batching, artifact, archive, GPU/XLA, and memory-growth gates pass |
| Statistically supported ranking | None; no stochastic candidate comparison was designed |
| Descriptive-only differences | Canary loss, acceptance, and timings only |
| Default readiness | Shared harness integration only; no model/filter/training default promoted |
| Next evidence | Per-cell P2 target freeze, value/score/reference admission, independent recomposition, then same-target comparator and target-specific training |

## Post-Run Red Team

The strongest alternative explanation is that the canary passes because an
analytic two-dimensional Gaussian is easy. That is true and is why numerical
quality is not the primary criterion. P1 establishes fail-closed integration
boundaries, not nonlinear transport performance. The mutation and substitution
tests, callable-source bindings, state guards, and artifact replay are the
relevant evidence.

The weakest remaining area is cross-process reconstruction: P1 issues and
replays typed identities within the executing process and persists their full
payload, but a dedicated loader for reconstructing an identity object in a new
process is not provided. Later phases should reconstruct a fresh identity from
the frozen contract/adapter/recomposition rather than trusting a deserialized
caller object. This is a limitation, not a P1 veto.

## Handoff

P2 begins at target repair, not at HMC or training. It must freeze serious data,
priors, charts/Jacobians, filter settings, batch-native adapters, and independent
recomposition for `SVX-SGQF` and `SVX-ZC`. Each cell remains independent, and
`SVX-ZC` additionally remains blocked on an admitted production fixed Zhao-Cui
source route. P1 can be reopened only for a common harness defect.

Bounded Claude read-only terminal review found no material defect and returned
`VERDICT: AGREE`; see
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p1-claude-review-record-2026-07-15.md`.

After that review, P2 target-adapter design exposed one common identity defect:
hard-veto `target_status_telemetry` was executed by transformed HMC but was not
source-bound in the typed identity. P1 was reopened under its common-repair
budget, added the third execution surface and mutation test, passed 72 focused
tests, and passed fresh attempt 06. This strengthens the reviewed claim and does
not change its scope or nonclaims.
