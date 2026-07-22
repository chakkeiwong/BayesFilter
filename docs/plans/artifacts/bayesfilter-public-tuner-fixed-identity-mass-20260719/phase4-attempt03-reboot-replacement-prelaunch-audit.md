# Phase 4 Attempt 03 Reboot-Replacement Prelaunch Audit

Snapshot time: `2026-07-19T23:49:33+08:00`

Status: `PASS_FOR_REBOOT_REPLACEMENT_LAUNCH`

## Scope And Classification

Attempt 03 is a one-for-one infrastructure replacement for the interrupted
Attempt 02 process. It does not add a third scientific seed, change the Phase 4
question, or reuse an incomplete runner root. Attempt 02 is preserved at
`phase4-lgssm-attempt02/` and is classified
`INTERRUPTED_BY_REBOOT_BEFORE_TUNING_ADMISSION`.

No terminal tuning result, replay-hash gate, warm-up archive, retained sample
archive, cell result, or run manifest exists for Attempt 02. Its durable events
are tuning-repair diagnostics only and provide no posterior or truth-tail
evidence. The historical PID `1580591` is absent after reboot.

## Evidence Contract

Question: does the second sequential sampling seed, using the exact preserved
LGSSM transport and the exact admitted Attempt 01 public-tuner kernel, pass
sampler validity and the owner's truth-tail diagnostic?

Baseline and fixed identities:

- target signature:
  `bd40a828bc4916e5e09a8e6135f315ebc45c06844aed38a506d6296c2642557d`;
- frozen transport SHA-256:
  `b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e`;
- public tuner seed: `(20260621, 8)`;
- expected final-kernel hash:
  `e46effed4649e4cb7c3e25343549ab4c22315269fc46ccdba7b6506c076077fc`;
- fixed identity mass signature:
  `25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe`;
- target acceptance `0.70`, admission band `[0.65, 0.75]`;
- sequential seed offset `1000`;
- GPU/XLA, TF32, and TensorFlow memory growth enabled;
- warm-up and retained caps of `10,000` draws per chain.

Primary criterion: deterministic public retuning must reproduce the exact
Attempt 01 final-kernel hash before sequential sampling. Sampling then must pass
the declared target-health, energy, movement, modern R-hat, bulk ESS, tail ESS,
and truth-tail gates.

Continuation vetoes: transport or target drift, final-kernel hash mismatch,
nonfinite target or gradient, mass mutation, missing required diagnostics,
GPU/XLA or memory-growth failure, corrupted artifacts, or exhaustion of the
unchanged Phase 4 replacement budget.

Explanatory only: tuning trajectory, selected step size and leapfrog count,
runtime, acceptance trajectory outside its explicit admission role, and
posterior summaries not named as gates.

Nonclaims: even a pass does not establish universal NeuTra validity, sampler
superiority, distributional equivalence, default readiness, or scientific
validity outside this limited LGSSM diagnostic.

## Skeptical Plan Audit

| Risk | Audit result |
| --- | --- |
| Wrong baseline | Passed: the comparator is Attempt 01's exact final-kernel hash, not merely a similar configuration. |
| Proxy promoted to criterion | Passed: local tests, tuning screens, and acceptance telemetry do not replace sequential validity, ESS, convergence, or truth-tail gates. |
| Hidden changed default | Passed: target, transport, tuner seed, mass policy, thresholds, hardware class, GPU/XLA policy, and scientific seed remain unchanged. |
| Stale or mismatched implementation | Passed: all six implementation/document hashes recorded by the reboot memo match exactly; Git remains `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`. |
| Unfair replication | Passed conditionally: the exact final-kernel hash gate must pass before sampling; otherwise classify `TUNING_REPLAY_HASH_MISMATCH`. |
| Missing stop condition | Passed: replay, target, mass, diagnostics, device policy, artifact validity, and budget vetoes are explicit. |
| Misleading successful command | Passed: a process exit alone is not evidence; interpretation requires the replay gate and terminal structured artifacts. |
| Environment mismatch | Passed: trusted preflight saw the RTX 4080 SUPER; TensorFlow 2.19.1 created `GPU:0`; the repository helper verified memory growth before logical-device initialization; and the test graph compiled with XLA. |
| Output collision or invalid resume | Passed: the Attempt 03 root was absent before launch and the runner will start in a fresh versioned root. |
| Budget expansion | Passed: the reboot killed Attempt 02 before tuning admission or sampling, so this localized infrastructure replacement uses the same remaining second-seed authorization. |

The plan survives skeptical audit. The earliest discriminating diagnostic is
the trusted device preflight, followed by the exact tuning replay-hash gate.
Do not interpret truth-tail evidence unless both pass and sequential sampling
completes validly.

## Prelaunch Integrity Checks

The following hashes match the reboot memo:

```text
9f418a1e5d952c85bdec648694a1cb8dea75dc1c506f5e18642e2ca7dcf4afe5  bayesfilter/inference/neutra_end_to_end.py
26490a68bf9e39cfc42dac01b861071884389402ebf7cde62b0af2926273813d  docs/benchmarks/run_neutra_all_models_end_to_end_2026_07_18.py
6fc75b865518af0beb36915660c1378d98ebb05e6d0b6ffd3b3a1ce7952ef9d2  tests/test_neutra_all_models_end_to_end_contract.py
027c367df698e8f41e11f0d8016a973e74698ddda29331fec654f10273591896  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase3-next-subplan.md
41cbb841f8531569461a8bbe75ae957b5a4838b513c683b78df2356881563b6b  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt01-result.md
e088bf4172038a3787ffc7cbd724dd96c2080220d85370eed1b7883babc2d057  docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-attempt02-prelaunch-audit.md
```

The preserved transport hash also matches exactly. Existing focused engineering
evidence remains applicable because these files are unchanged: `22 passed` for
the frozen-validation contract suite, `134 passed, 1 skipped` for the broader
public-tuner/fixed-mass/replay/NeuTra/public-API regression, plus passing
`py_compile` and focused `git diff --check`.

Trusted preflight completed at `2026-07-19T23:50:58+08:00`:

- `nvidia-smi` reported NVIDIA GeForce RTX 4080 SUPER, driver `591.86`, and
  CUDA `13.1`;
- the TensorFlow GPU device was
  `/job:localhost/replica:0/task:0/device:GPU:0`;
- memory policy schema was `bayesfilter.tensorflow.gpu_memory_policy.v1` in
  `memory_growth` mode with `memory_growth=true` for every physical GPU;
- `configured_before_logical_device_initialization=true` and
  `full_device_preallocation_disabled=true`;
- a matrix operation compiled and executed under XLA on the GPU.

## Exact Launch

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
