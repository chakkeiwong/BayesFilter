# Phase 4 NeuTra LGSSM Reboot Reset Memo

Snapshot time: `2026-07-19T22:36:30+08:00`

Status: `ACTIVE_ATTEMPT_NOT_REBOOT_RESUMABLE`

Read this memo first after a reboot. It supersedes conversational state for the
active Phase 4 second-seed replication, but it does not supersede the master
plan or preserved result artifacts.

## Reboot Verdict

A reboot is **safe for the repository, plans, code changes, preserved
artifacts, and scientific recovery information**. It is **not lossless for the
currently running computation**.

At the snapshot time, host PID `1580591` was still running on the trusted GPU:

```text
/home/chakwong/anaconda3/envs/tf-gpu/bin/python
docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py
--action validate-frozen --cell LGSSM-EXACT ... --seed-offset 1000
```

The process had run for about two hours. A reboot will terminate it. No
terminal public tuning result, replay-hash gate, sequential sample archive, cell
result, or run manifest had been emitted. Sequential sampling had therefore
not been authorized and no Attempt 02 truth-tail evidence exists.

The top-level runner is not reboot-resumable from the existing output root:
`run_neutra_frozen_transport_validation_cell` requires a fresh cell root. The
public progress artifact reports an in-process private Phase 7 repair handoff,
but that private payload is not exposed as a disk-backed runner resume entry and
will be lost when the process exits. Do not try to relaunch into Attempt 02.

Recovery is nevertheless complete: preserve Attempt 02 as an interrupted
infrastructure artifact and launch one replacement attempt in a fresh Attempt
03 root under the unchanged Phase 4 scientific contract. Under the repository
campaign-repair policy, this replaces the interrupted second-seed launch; it
does not add a third scientific seed or change the campaign question.

## Scientific Contract

Question: does a second sequential sampling seed, using the exact same
preserved LGSSM transport and exact same admitted public-tuner kernel as Phase
4 Attempt 01, pass sampler validity and the owner's truth-tail diagnostic?

Required invariants:

- cell: `LGSSM-EXACT`;
- target signature:
  `bd40a828bc4916e5e09a8e6135f315ebc45c06844aed38a506d6296c2642557d`;
- frozen transport SHA-256:
  `b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e`;
- public tuner seed: `(20260621, 8)`;
- expected final-kernel hash:
  `e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`;
- fixed identity mass signature:
  `25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe`;
- target acceptance `0.70`, valid band `[0.65, 0.75]`;
- sequential seed offset `1000` only;
- GPU/XLA, TF32, and TensorFlow memory growth enabled;
- warm-up and retained adaptive caps remain `10,000` draws per chain.

Attempt 03 counts as a pure sampling-seed replication only if deterministic
retuning reproduces the exact expected final-kernel hash before sampling. On a
mismatch, the runner must stop with `TUNING_REPLAY_HASH_MISMATCH`; the run must
not be interpreted as second-seed sampling evidence.

## Completed Work Before Snapshot

The fixed-identity public tuner migration is complete:

- public `tune_hmc_kernel` supports `mass_policy="fixed_identity"`;
- active NeuTra orchestration uses the public tuner and BayesFilter replay
  adapter;
- an optional exact final-kernel hash gate was added to preserved-transport
  validation;
- CLI option `--expected-tuning-final-kernel-hash` forwards the gate;
- `seed_offset` changes only sequential warm-up and retained seeds;
- a mismatch is recorded before sequential sampling is authorized.

Local engineering evidence:

- frozen-validation contract suite: `22 passed`;
- public tuner/fixed-mass/replay/NeuTra/public-API focused regression:
  `134 passed, 1 skipped`;
- `py_compile`: passed;
- focused `git diff --check`: passed;
- bounded Claude review of the revised Attempt 01 result:
  `VERDICT: AGREE`.

Trusted preflight evidence:

- NVIDIA RTX 4080 SUPER visible;
- TensorFlow `2.19.1` saw `/physical_device:GPU:0`;
- BayesFilter memory policy reported memory growth enabled and full-device
  preallocation disabled;
- a tiny TensorFlow operation compiled and executed with XLA on GPU.

## Attempt 01 Preserved Result

Path:
`docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt01/LGSSM-EXACT/`

Decision: `MARGINAL_RERUN`.

- final-kernel hash:
  `e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`;
- tuning verification acceptance: `0.7109046802`;
- warm-up: 2,000 draws/chain, max modern R-hat `1.0114386437`;
- retained: 1,000 draws/chain, max R-hat `1.0094533522`;
- minimum bulk ESS `1083.28598`;
- minimum tail ESS `1192.15611`;
- retained acceptance `0.691`;
- zero energy divergences and valid target telemetry;
- only `q2` was marginal, `p_truth=0.0457385654`, above the severe threshold
  `0.003`.

Owner policy therefore requires one second sampling seed with the same kernel.

## Interrupted Attempt 02

Preserve this root exactly; never delete, overwrite, or relaunch into it:

`docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt02/`

Last durable public progress snapshot:

```text
timestamp_utc: 2026-07-19T14:33:05.002052Z
process_id: 1580591
phase7_last_attempt_index: 1
current_stage: fixed_mass_ladder_repair_screen_call_start
last_completed_stage: fixed_mass_ladder_screen_call_complete
round_kind: final_local
candidate_count: 4
candidate_completed_count: 3
candidate_pass_count: 2
candidate_hard_veto_count: 0
selected_pair_exists: true
mass_hash: 25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe
```

Last durable private event before the snapshot: outer attempt `1` completed
its windowed fixed-identity mass stage at
`2026-07-19T14:26:32.347069Z`; final status was `passed` and the mass signature
remained unchanged.

The previous outer attempt selected step size `0.919034967924014` and four
leapfrog steps, then produced an acceptance-inconclusive verification and
entered the native retry. That is tuner repair evidence only, not a final
kernel and not posterior evidence.

Files present in Attempt 02 include:

- `frozen_transport_input.json`;
- `tuning/hmc_kernel_tuning_progress.json`;
- `tuning/private_diagnostics/hmc_tuning_events.jsonl`;
- initial and per-attempt mass `.npz` artifacts.

Files absent at the snapshot:

- `tuning/hmc_kernel_tuning_result.json`;
- `tuning_replay_hash_gate.json`;
- `result.json`;
- `run_manifest.json`;
- all warm-up and retained sample archives.

Classification after reboot: `INTERRUPTED_BY_REBOOT_BEFORE_TUNING_ADMISSION`.
Do not classify this as tuning failure, sampling failure, truth-tail failure,
or evidence against NeuTra.

## Recovery Procedure

1. Confirm the old process is absent. PID `1580591` is historical after reboot.
2. Read this memo, the Phase 4 subplan, Attempt 01 result, and prelaunch audit.
3. Confirm the implementation files still match the hashes below or inspect
   any intentional change before running.
4. Run trusted `nvidia-smi` and the BayesFilter TensorFlow GPU/XLA memory-growth
   preflight.
5. Confirm the Attempt 03 root does not exist.
6. Launch the exact Attempt 03 command below with trusted/escalated GPU access.
7. Do not interpret anything until `tuning_replay_hash_gate.json` exists and
   has status `PASS`.
8. After terminal completion, write the Phase 4 result, terminal review, reset
   memo update, and Phase 5 handoff only if the stated gates permit it.

Implementation SHA-256 snapshot:

```text
9f418a1e5d952c85bdec648694a1cb8dea75dc1c506f5e18642e2ca7dcf4afe5  bayesfilter/inference/neutra_end_to_end.py
26490a68bf9e39cfc42dac01b861071884389402ebf7cde62b0af2926273813d  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py
6fc75b865518af0beb36915660c1378d98ebb05e6d0b6ffd3b3a1ce7952ef9d2  tests/test_neutra_all_models_end_to_end_contract.py
027c367df698e8f41e11f0d8016a973e74698ddda29331fec654f10273591896  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase3-next-subplan.md
41cbb841f8531569461a8bbe75ae957b5a4838b513c683b78df2356881563b6b  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt01-result.md
e088bf4172038a3787ffc7cbd724dd96c2080220d85370eed1b7883babc2d057  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt02-prelaunch-audit.md
```

Snapshot Git commit: `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`.

The listed lane files are currently untracked in the shared dirty worktree.
They are preserved by a normal reboot because they are on disk, but they are
not protected by Git history. Do not run cleanup, reset, checkout, or restore
commands against them.

### Trusted preflight

```bash
nvidia-smi
```

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python -c \
  'import json, tensorflow as tf; from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth; p=configure_tensorflow_gpu_memory_growth(tf, require_gpu=True); tf.config.set_soft_device_placement(False); f=tf.function(lambda x: tf.linalg.matmul(x,x), jit_compile=True); y=f(tf.eye(2)); print(json.dumps({"policy": p, "device": y.device, "value": y.numpy().tolist()}))'
```

### Attempt 03 replacement launch

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py \
  --action validate-frozen \
  --cell LGSSM-EXACT \
  --output-root docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt03-reboot-replacement \
  --frozen-transport docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02/LGSSM-EXACT/final/segments/steps-004001-005000/frozen_transport.json \
  --frozen-transport-sha256 b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e \
  --expected-tuning-final-kernel-hash e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc \
  --seed-offset 1000
```

## Interpretation After Recovery

- Exact kernel hash mismatch: stop before sampling; classify as
  `TUNING_REPLAY_HASH_MISMATCH`, not a second-seed result.
- Sampler hard veto or convergence/ESS failure: Phase 4 negative sampler
  result; do not evaluate truth-tail promotion.
- Valid sampling and every `p_truth >= 0.05`: limited second-seed success under
  the owner's rule.
- Valid sampling with the same `q2` failure again: concerning repeated failure;
  investigate rather than declaring success.
- Valid sampling with a different marginal failure in `[0.003, 0.05)`: report
  the two-seed pattern directly; do not invent another seed without a refreshed
  plan and user direction.
- Any `p_truth < 0.003`: severe failure requiring investigation.

Even a pass establishes only the limited LGSSM diagnostic claim. It does not
establish universal NeuTra validity, sampler superiority, distributional
equivalence, or default readiness.

## Source Documents

- Master plan:
  `docs/plans/bayesfilter-public-tuner-fixed-identity-mass-plan-2026-07-19.md`
- Active Phase 4 subplan:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase3-next-subplan.md`
- Attempt 01 result:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt01-result.md`
- Attempt 02 prelaunch audit:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt02-prelaunch-audit.md`
- Attempt 02 interrupted artifacts:
  `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt02/`
