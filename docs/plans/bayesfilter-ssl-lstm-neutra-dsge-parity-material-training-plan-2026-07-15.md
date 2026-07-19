# SSL-LSTM NeuTra DSGE-Parity Material Training Plan

Date: 2026-07-15

Status: `COMPLETED_SEED_INSTABILITY_REPAIR_REQUIRED`

Completion update, 2026-07-15: both authorized seeds completed within the
10-hour shared cap. Seed A passed every nomination gate; seed B had no hard
veto but failed the moderate-shell and scale-saturation promotion screens.
The prospective program decision is `SEED_INSTABILITY_REPAIR_REQUIRED`.
Details and immutable hashes are in
`bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-result-2026-07-15.md`.
No Phase 5, HMC, or repair run is authorized by this closeout.

Pre-launch review update, 2026-07-15: bounded Claude review of this exact plan
returned `VERDICT: AGREE` with no material findings. Claude successfully read
the dedicated runner in three subsequent bounded attempts, but the response
stream repeatedly stalled with API retries and produced no implementation
verdict; those attempts are not counted as approval. A focused native
implementation audit found and repaired three material defects before launch:
resource stops now preserve the exact latest state, program wall time is
assigned before final classification, and actual per-seed/shared-cap overruns
are hard vetoes. The repaired CPU-hidden gate is `52 passed`; compilation and
`git diff --check` pass. The implementation review status is
`NATIVE_AGREE_AFTER_REPAIR_CLAUDE_TRANSPORT_NO_VERDICT`.

Owner authorization, 2026-07-15: approve the proposed **10 trusted GPU-hour
contingency cap** for two independent 5,000-step source-parity preset seeds
with sequential stopping.

## 1. Research Intent And Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Main question | Can two independently initialized and trained instances of the source-matched local Rotemberg/SGU plain-NeuTra procedure produce finite, replayable, sufficiently broad frozen transports on the locked SSL-LSTM target? |
| Candidate mechanism | The frozen `dsge_paper_dense_iaf` preset: three `(4,4)` ELU dense IAF stages, reverse-coordinate mixing between stages, fixed identity-scale translation to `PRIOR_CENTER_VALUES`, reverse KL, batch 480, 5,000 steps, Adam `0.01` with epsilon `1e-7`, schedule boundaries `[999,3999]`, per-variable norm clipping at 10, and score matching off. |
| Exact target | Locked identity-oriented four-coordinate SSL-LSTM SVD-UKF target with semantic SHA-256 `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`. |
| Exact baseline/comparator | Each seed's untrained instance under the same topology, target, fixed validation rows, and initialization. Historical affine A/B remain classical controls for later matched HMC/predictive work; the old one-stage `tanh` arm is a negative ablation and is not the plain nonlinear baseline. No method ranking occurs in this phase. |
| Primary candidate nomination criterion | A completed seed has no hard veto or promotion veto: exact source/target/config binding, finite updates and validation, exact early resume replay, one-sided paired heldout final-minus-initial loss upper bound below zero, original-neighborhood and moderate-shell inverse radii at most `4.30`, dense scale saturation fraction at most `0.05`, frozen roundtrip maximum at most `1e-9`, finite frozen transformed scores, exact serialization/reload, GPU residency, and XLA execution. |
| Phase pass | Both prospectively independent seeds complete and are nominated. One pass plus one rejection is seed-instability evidence and triggers a repair plan; two candidate rejections reject this exact training candidate under these gates, not NeuTra generally. |
| Hard evidence veto | Source or sibling-commit drift; target/chart/signature drift; seed overlap; nonfinite target, score, loss, gradient, update, optimizer state, validation, frozen map, or probe; exact-resume mismatch; corrupt/mutating restore; serialization/reload mismatch; CPU fallback; missing XLA/GPU evidence; invalid JSON/artifact; or shared-budget overrun. |
| Promotion veto | Heldout trainer gate fails; original-neighborhood or moderate-shell support screen fails; saturation exceeds the cap; roundtrip exceeds tolerance; or frozen score/reload gate fails without invalidating the harness. |
| Continuation veto | A hard evidence veto that invalidates the target, harness, implementation, source binding, runtime, or artifacts; exhaustion of the shared cap; or insufficient remaining cap to give the next seed its prospectively bounded complete attempt. A promotion veto for seed A is **not** a continuation veto and does not suppress seed B. |
| Repair trigger | One nominated and one rejected seed; both finite seeds rejected for saturation/support/heldout behavior; reproducible clipping pathology; or external interruption after a valid checkpoint. |
| Explanatory only | Training and validation loss curves, gradient norms, clipping frequency, parameter norms, compilation time, checkpoint time, prior/far-tail inverse radii, and continuous differences between A and B. |
| Nonclaims | No posterior correctness, complete mode/tail coverage, HMC readiness, sampler convergence, predictive equivalence, statistical superiority, paper fidelity generally, default readiness, or rejection/success of NeuTra as a research direction. |
| Structured artifacts | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training/` |
| Result artifact | `docs/plans/bayesfilter-ssl-lstm-neutra-dsge-parity-material-training-result-2026-07-15.md` |

Diagnostic roles are fixed prospectively. Nonfinite state, identity drift,
resume/reload failure, and GPU/XLA failure are hard evidence vetoes. The
support, saturation, heldout, roundtrip, and frozen-score screens are candidate
promotion vetoes unless they reveal corrupted computation. Loss values and all
continuous A/B differences are explanatory; they cannot rank the seeds.

## 2. Independent Seed And Artifact Contract

| Role | Seed A | Seed B |
| --- | --- | --- |
| Initialization | `[20260715,4101]` | `[20260715,4102]` |
| Training base stream | `[20260715,5101]`, folded by logical step | `[20260715,5102]`, folded by logical step |
| Fixed heldout base rows | `[20260715,5201]` | `[20260715,5202]` |
| Candidate directory | `seed-a/` | `seed-b/` |

The timing-canary seed `[20260715,4099]`, historical training/validation
roles, and fixed prior-probe seed `[20260714,3301]` are disjoint from both
material seed namespaces. The prior-probe rows are a shared deterministic
diagnostic bank, not training or heldout rows.

Each candidate writes immutable numbered checkpoints every 100 steps, fixed
heldout validation every 250 steps, a final trainer state, a six-component
frozen payload, and a structured result. An exact restore-and-replay check is
performed near the start before material evidence accumulates. Existing
nonempty output directories fail closed. External interruption may be repaired
only by an explicitly recorded exact checkpoint continuation with the same
seed/config/stream and source identities; it may not create a fresh replacement
seed under the same label.

## 3. Candidate Gates

The gate definitions are inherited from the corrected historical Phase 4 plan
so this source-matched run answers the intended counterfactual rather than
changing the screen after observing the one-stage failure:

1. Historical A4 starts and their `+/-0.10` coordinate neighborhoods, in the
   locked A0 geometry, must each invert to finite standard-normal radius at
   most `4.30`.
2. The eight locked `+/-2.0` moderate-shell directions must each invert to
   finite standard-normal radius at most `4.30`.
3. The `+/-4.0` far-tail directions and 16 prior probes remain explanatory
   radius diagnostics. Their value, score, map, inverse, and logdet must remain
   finite.
4. Across the fixed heldout rows and all three IAF stages, the fraction of
   scale logs satisfying `abs(scale_log) >= 0.95*s_max` must be at most `0.05`.
5. The paired heldout final-minus-initial per-row loss one-sided 95% upper
   bound must be below zero. This is a candidate trainer veto, not posterior
   evidence or a seed-ranking statistic.
6. Frozen forward/inverse roundtrip maximum absolute error must be at most
   `1e-9`; target and transformed scores, values, and log determinants must be
   finite; serialized reload must preserve the procedure, chart, signatures,
   component order, tensors, and training-state hash.

A and B are independent replications of one fixed candidate, not competing
hyperparameters. Passing a screen means viable for exact transformed-target
preflight; it does not establish statistical superiority or correctness.

## 4. Resource Contract And Sequential Stopping

The shared cap is exactly `36,000` charged trusted GPU-seconds. Each seed has a
maximum `18,000`-second complete-attempt allowance including compilation,
training, validation, checkpointing, freezing, and probes. The authoritative
canary estimated `15,042.7` training-step seconds per seed; the per-seed cap
therefore leaves about 19.7% contingency for non-step work and timing
variation. This is a cap, not a completion guarantee.

Run seed A, then seed B. Seed B starts after a finite candidate rejection of A
because replication is necessary to interpret that rejection. Do not start B
after a hard evidence veto, or if A consumed enough time that B cannot receive
its full prospective attempt inside the shared cap. Stop a running seed before
either its 18,000-second allowance or the shared 36,000-second cap is crossed,
write its latest state and failure receipt, and do not call a truncated run a
candidate.

No repair topology, learning-rate variation, extra steps, HMC, forecasting, or
third seed is authorized by this plan. Unused budget does not authorize scope
expansion.

## 5. Skeptical Pre-Execution Audit And Pre-Mortem

Audit status: `PASS_AFTER_REPLACING_HISTORICAL_RUNNER_WITH_DEDICATED_HARNESS`.

| Audit item | Finding and control |
| --- | --- |
| Wrong baseline | The historical runner hard-codes the unfaithful one-stage `tanh`, batch-64, constant-LR arm. It must not be reused as the material candidate. A dedicated runner must call only `dsge_paper_neutra_config` and reject override knobs. |
| Proxy promoted | Heldout loss is only a candidate trainer veto; support probes are nomination screens; timing is only a resource diagnostic. None establishes posterior correctness or ranks A/B. |
| Missing stop | Per-seed and shared wall/GPU-second caps, hard vetoes, promotion vetoes, and the A-to-B continuation rule are explicit. |
| Unfair comparison | A and B use identical procedure/target/budget and disjoint fixed seeds. Historical affine controls are preserved for later matched comparisons and are not ranked here. |
| Hidden assumption | The procedure operates directly in the locked identity chart; the fixed translation is the prior center; source parity is with local `dsge_hmc` commit `d94566c9f70b3143e599a56eba7cb461ff2bda88`, not the literature generally. |
| Stale context | The final-source canary and 43 focused tests bind trainer, loader, target, runner, and direct parity source hashes. The dedicated runner must fail closed if those implementation identities drift before launch. |
| Environment mismatch | Serious execution is TensorFlow `float64`, trusted GPU, XLA JIT, TF32 enabled, soft placement disabled. CPU-hidden tests are engineering checks only. |
| Artifact answers question | Per-seed receipts preserve exact config, seeds, histories, checkpoints, final/frozen hashes, gate decisions, device/JIT provenance, charged time, source identities, and nonclaims. |

Pre-mortem: the runs could lower reverse KL while mode-seeking; fixed
neighborhood/shell screens and later HMC/predictive gates prevent that loss
from becoming posterior evidence. A could fail from initialization noise; B
still runs after a promotion veto. Both could pass the finite probe bank while
missing unknown modes; no completeness claim is allowed. A process or editor
could crash; execution must be detached from the editor and checkpoints must
permit a separately recorded exact continuation. A first run could use nearly
all the budget and leave a truncated B; the five-hour per-seed ceiling prevents
that. A candidate could pass while the local source transfer is wrong; direct
cross-repository parity and source-hash checks are entry conditions.

## 6. Implementation, Review, And Commands

Implement a dedicated runner and focused contract tests. Reuse the corrected
probe definitions, but do not mutate the historical Phase 4 artifacts or
reinterpret their results. Required CPU-hidden checks:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_neutra_reverse_kl_training.py \
  tests/test_neutra_dsge_procedure_parity.py \
  tests/test_dense_iaf_neutra_artifact_loader.py \
  tests/test_ssl_lstm_neutra_dsge_parity_material_training.py
```

One focused read-only review must verify exact preset use, seed separation,
budget arithmetic, A-to-B continuation semantics, mutation/reload checks, and
claim boundaries. Repair material findings and rerun focused checks before GPU
launch.

The serious program command is frozen as:

```bash
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_neutra_dsge_parity_material_training_2026_07_15.py \
  --program-output-root \
  docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/dsge-parity-material-training
```

Launch it as a detached user service so editor/session failure does not kill
the scientific run. Record unit name, PID, command, environment, device, start
time, and journal inspection command. Do not launch if another process is using
the selected GPU materially.

## 7. Handoff

After both candidate attempts or a continuation veto, write the result note
with the run manifest, decision table, inference-status table, per-seed gate
table, budget accounting, strongest alternative explanation, and next smallest
discriminating action.

- Two nominated seeds: proceed to an exact frozen transformed-target preflight
  plan for both; do not select a descriptively favorable seed.
- One nominated seed: stop for a seed-instability repair design; do not promote
  the lone seed to the main HMC lane.
- Two finite rejected seeds: reject this exact source-matched training
  candidate under the declared gates and diagnose the shared failure before
  altering topology or objective.
- Hard evidence veto: identify whether it invalidated the harness,
  implementation, target, runtime, or artifact, repair that layer, and do not
  interpret candidate quality.

The research direction is not rejected by a candidate promotion veto. No next
phase begins automatically under this authorization.
