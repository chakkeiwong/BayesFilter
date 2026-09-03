# SSL-LSTM q=20 Phase 9A chart-1/beta-0 program-repair result

Date: 2026-09-01  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-subplan-2026-09-01.md`  
Status: `PASS_LOCALIZED_MECHANICS_ONLY_FULL_REPLAY_PENDING`

## Verdict

The executable repair is successful for the bounded chart-1/beta-0
localization question. The repaired runner rebuilt both charts from a fresh
v4 seed namespace, measured all four declared `(epsilon, L)` pairs for
chart-1/beta-0, performed two independent selection replications, performed
disjoint held-out verification, and wrote a durable fixed-transport handoff.
The final fresh attempt completed on GPU0 in `494.5689085649792` seconds with
verified TensorFlow memory growth and a 1,402,670,592-byte peak allocator
reading below the 4,294,967,296-byte cap.

This is a program and mechanics result, not a trained-NeuTra or posterior
result. The selected handoff has `epsilon=0.55`, `L=3`, and
`authority_status=tuning_candidate`. The selection and held-out schedules have
only four retained draws per chain. Acceptance is high (selection means
`0.923473` for the selected `(0.55,3)` candidate and `0.941270` for `(0.55,8)`;
the selected candidate's replication values were `0.915687` and `0.931260`;
held-out `0.941146`), so
the guide's acceptance-band repair trigger fired. That trigger is evidence that
the next tuning run needs more resolution or a different scale; it is not
evidence of convergence or good tuning.

## Attempt history

| Attempt | Profile | Outcome | Interpretation |
|---|---|---|---|
| `attempt-01` | `chart1_beta0_repair_v1` | Outer wall bound reached before a candidate completed | Resource/execution failure; no candidate conclusion |
| `attempt-02` | `chart1_beta0_repair_v2_bounded` | Signal 15 after entering scope, before first full-chain candidate | Resource/execution failure; durable failure and start manifests preserved |
| `attempt-03` | `chart1_beta0_repair_v3_minimal` | Completed | Fresh mechanics probe, but the manifest pointed to the historical result note |
| `attempt-04` | `chart1_beta0_repair_v3_minimal` | Completed | Deterministic provenance replay; reused attempt-03 seeds and is not independent evidence |
| `attempt-05` | `chart1_beta0_repair_v4_fresh` | Completed | Final fresh-seed mechanics evidence; used below |

Attempts 01--05 are immutable and live under
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-01/phase9a-chart1-beta0-repair/`.
The timeout attempts are not treated as failed tuning candidates and were not
used to warm-start attempt-05. Attempt-04 intentionally reused attempt-03's
v3 seeds to isolate a metadata patch; it verifies provenance behavior only and
does not satisfy the fresh-seed requirement.

## Evidence contract and observed evidence

| Question | Required evidence | Observed result | Role and limit |
|---|---|---|---|
| Did the launch use the intended target? | Exact target and backend signatures | Target `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`; backend `tensorflow_eigh_strict`; C5 freeze `phase8-k2-compact-high-l3-pure` | Hard identity check passed; it does not validate the learned map |
| Is the bridge finite and proper for this preflight? | Finite value/score/status and properness receipt | All beta preflight values and scores finite; properness receipt reports positive likelihood and Gaussian innovation factorization | Hard mechanics check passed; it is not a mode-discovery or posterior check |
| Were charts fresh and replayable? | Fresh checkpoint hashes, optimizer-free restore, one training trace per update graph | Six fresh chart-level checkpoints (two charts, three betas); trace count `1` at every level; restore state hashes matched | Chart mechanics passed; pullback residuals remain large |
| Is transport bookkeeping reliable? | Round-trip/log-determinant and finite physical-score checks | Reliability receipt passed; self/declared round-trip maxima at or below `8.8818e-16`; cross log-determinant residuals were zero | Representation bookkeeping passed; this is not IID Gaussian whitening |
| Was tuning measured rather than inferred? | Every declared joint pair attempted and measured | Four of four pairs measured: `(0.55,3)`, `(0.55,8)`, `(1.2,3)`, `(1.2,8)`; directional inference and early stop were false | Hard provenance check passed |
| Did at least one fixed kernel remain mobile and finite? | Finite target/score/status, movement in every retained chain, no hard veto for selected candidate | All four pairs were eligible and had no hard veto in the fresh v4 run; all chains moved | Candidate screen passed for a localized handoff; native divergences were not exposed |
| Did held-out verification pass? | Disjoint held-out fixed-kernel run with finite values and movement | Selected `(0.55,3)` held-out status `passed`, all chains moved, no hard veto | Tiny mechanics screen only; held-out acceptance repair trigger fired |
| Can the Phase 9A transition run? | Six scope-specific handoffs | Only scope index `3` (chart-1/beta-0) was selected | Deliberately not run; status is `PASS_PHASE9A_SCOPE_PREFLIGHT_PARTIAL` |

### Candidate detail

The measured-grid policy was `measured_joint_grid_v1`, with acceptance band
`[0.45, 0.90]`, repair band `[0.30, 0.95]`, identity z-mass, two selection
replications, and four retained draws per chain. Selection used
`replicated_min_bulk_ess_per_gradient` only after all pairs were measured.

| Pair | Screen acceptance | Selection result | Hard veto | Interpretation |
|---|---:|---|---|---|
| `(0.55, 3)` | `0.947915` | Eligible; selected candidate index `0` | None | Viable mechanics candidate; high-acceptance repair trigger |
| `(0.55, 8)` | `0.962112` | Eligible | None | Viable mechanics candidate; high-acceptance repair trigger |
| `(1.20, 3)` | `0.788106` | Eligible | None | Viable mechanics candidate in this fresh seed realization |
| `(1.20, 8)` | `0.584719` | Eligible | None | Viable mechanics candidate in this fresh seed realization |

The selected candidate's two selection replications had acceptance means
`0.915687` and `0.931260`, minimum bulk ESS per declared gradient values
`0.200545` and `0.226287`; the maximum rank-normalized R-hat across the two
replications was `1.6158853` (the individual maxima were `1.6158853` and
`1.3068534`). These values are descriptive because the run has
four retained draws per chain and no inferential uncertainty analysis. The
candidate-selection object reports `all_candidate_pairs_measured=true` and no
directional inference. The fresh v4 outcome differs from the reused-seed v3
provenance replay, demonstrating that this short schedule is seed-sensitive.

The fresh-map pullback diagnostic reported centered log-density RMS values
`372.226799` (chart-0) and `380.853358` (chart-1). Those are large residuals,
consistent with the earlier observation that the transport is not yet close to
an IID standard Gaussian law. The round-trip checks passing only establishes
that the map and log-determinant implementation are internally consistent.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept runner repair for localized mechanics | Fresh measured grid, durable artifacts, finite/mobile candidate, held-out record | No target, bridge, memory, route-scan, artifact, or selected-candidate veto in v4 | Tiny fixed-kernel sample sizes, seed sensitivity, and repeated trainer construction | Preserve attempt-05 and design a separately audited six-scope replay/performance run | No default tuning policy |
| Retain `(epsilon=0.55,L=3)` as a provisional handoff | At least one eligible measured candidate and held-out finite/movement check | All four pairs were eligible, but all high-acceptance candidates triggered repair diagnostics | No statistically defensible ranking; v3 and v4 seed realizations differ | Re-tune with a target-specific budget and enough draws before any claim run | No convergence, posterior, or sampler-superiority claim |
| Keep Phase 9B closed | Six independent scope handoffs required | Five scopes were intentionally not run | Full-scope runtime and chart-specific curvature remain unknown | Open only after a new plan passes skeptical audit | No whitening, mode discovery, or HMC readiness |

## Inference status

| Evidence class | Status | Explanation |
|---|---|---|
| Hard veto screen | Passed for the localized v4 mechanics route; no candidate or held-out hard veto | All four candidates and the held-out run were finite and mobile; native divergence telemetry was unavailable and is recorded as such |
| Statistically supported ranking | None | Two replications and four draws per chain do not support a ranking |
| Descriptive-only differences | High acceptance, ESS/R-hat values, seed-to-seed candidate changes, elapsed times, and pullback residuals | These nominate repairs and explain behavior; they are not inferential evidence |
| Default-readiness | Not ready | The artifact is explicitly `tuning_candidate` and `posterior_ready=false` |
| Next evidence needed | Fresh six-scope measured replay with larger target-specific evidence and performance accounting | It must preserve disjoint tuning/claim data and the same identity checks |

## Program repairs made

The repaired runner now has source-owned launch profiles, an immutable scope
pin for chart-1/beta-0, fresh seed namespaces, explicit joint-grid accounting,
per-call start/complete/failure records, pre-import start manifests, signal
failure handling, and a non-overwriting shell timeout fallback. The wrapper
sets `CUDA_VISIBLE_DEVICES=0` and `TF_FORCE_GPU_ALLOW_GROWTH=true` before the
TensorFlow import. The final manifest exposes the profile ID at the top level
and points to this result note. Focused tests cover profile pinning, grid
cardinality, seed freshness, failure durability, wrapper flags, and fail-closed
GPU environment checks.

The run emitted TensorFlow trainer-retracing warnings while constructing fresh
trainer objects for separate chart/beta levels. The per-trainer training graph
trace checks still reported one trace, so the warning did not invalidate this
mechanics artifact. It is a real performance risk for the larger replay and
must be measured or repaired before any production/default interpretation.

## Exact run record

- Command: `BAYESFILTER_PHASE9A_ATTEMPT_ID=attempt-05 bash scripts/run_ssl_lstm_q20_phase9a_chart1_beta0_repair_gpu.sh`
- Python: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` in conda environment `tfgpu`
- TensorFlow: `2.20.0`; TensorFlow Probability route; XLA enabled; TF32 enabled
- Device: one visible logical `/device:GPU:0`, NVIDIA GeForce RTX 4080 SUPER
- Memory policy: `memory_growth`, configured before logical-device initialization on every visible physical GPU; full-device preallocation disabled
- Allocator: current `623104`, peak `1402670592`, cap `4294967296` bytes
- Scope: chart index `1`, beta `0.0`, scope index `3`; profile `chart1_beta0_repair_v4_fresh`
- Seeds: recorded in the manifest under `seed_ledger`; all v4 roots are distinct from attempts 01--04
- Git commit: `54201f5cd925ed15036bad8156606b812d53b045`; worktree was already dirty; status hash `f2930627b70369b356ac2165e8e39e23688eb534e245e64b9d1f2fe3aba8293b`
- Wall time: `494.5689085649792` seconds
- Manifest hash: `88bab1482475d59d1c610f1dda4391d8b1424f15b69536895995950e327dfea7`
- Executed-plan snapshot hash in the manifest: `e744732e71fca22f8d1ae136b1870edffe5d7ce5cee934d68a315ba9a7e98e2c`

Primary artifact:
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-01/phase9a-chart1-beta0-repair/attempt-05/run_manifest.json`

The companion tuning artifact is:
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-01/phase9a-chart1-beta0-repair/attempt-05/chart-1/beta-0/fixed_transport_hmc_tuning_result.json`

## Post-run red team

The strongest alternative explanation is that the apparent handoff is an
artifact of an extremely short chain and a permissive mechanics-only screen,
not a well-tuned kernel. This is supported by the high acceptance repair
triggers, large short-chain R-hat values, and the fact that the reused-seed v3
replay rejected both `epsilon=1.2` pairs while fresh-seed v4 accepted them.
A second concern is repeated TensorFlow tracing increasing the cost of a full
six-scope replay. The result would be overturned as a runner repair if a fresh
replay failed to measure all pairs, lost checkpoint identity, or exceeded the
allocator/memory-growth contract. The weakest evidence is the numerical quality
and seed stability of the selected tuning values; that question remains open.

## Next action

Do not open Phase 9B. Draft and separately audit a full Phase 9A replay or a
performance-focused subplan that (1) keeps the measured-joint-grid semantics,
(2) increases evidence only under a bounded budget, (3) addresses trainer
retracing, and (4) requires all six scope handoffs before the shared transition
controller. Attempts 04 and 05 may be used only as mechanics-localization
inputs, not as claim-run tuning artifacts.
