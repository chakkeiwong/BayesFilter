# Phase 2 GPU/XLA Preflight Result

Campaign: `bayesfilter-neutra-all-executable-models-e2e-20260718`

Date: 2026-07-18

Status: `PASS_TO_SERIOUS_CAMPAIGN`

## Result

The trusted device probe found one NVIDIA GeForce RTX 4080 SUPER with 16,376
MiB total memory. TensorFlow saw one physical and one logical GPU, configured
memory growth before logical-device initialization, disabled full-device
preallocation, enabled TF32, and created `/GPU:0` with 13,495 MiB available.

The active preflight artifact is
`docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/preflight-attempt-05/PP-UKF/result.json`.
It passed after 142.48 seconds.

The smoke used the current direct PP-UKF target signature, one fresh
`source_width_lr1e3` training step, batch size 128, GPU placement, XLA, one
compiled `tf.while_loop` invocation, the frozen transport loader, and compiled
frozen/trainable transport, log-Jacobian, pullback, and log-Jacobian-score
parity. Target values were finite, target status was valid, no scalar fallback
or Python sample-axis loop was used, and all training/optimizer/output tensors
were on GPU.

The repository native fixed-transport tuner was exercised through its supported
injected-runner seam. The resulting artifact bound the current target and
frozen transport, issued fixed identity mass in `z`, used target acceptance
`0.70`, kept both fixed-grid fields empty, and reached the tuner selection and
artifact path. The injected chain is an engineering contract smoke only; it is
not real HMC, convergence, or kernel-admission evidence. Phase 3 retains the
unchanged real native TFP runner and full fresh verification.

## Attempt classification

Attempts 01 through 04 are preserved partial preflight artifacts. They were
observed through foreground tool windows that ended before the long PP-UKF
compile/evaluation completed. The supervised attempt 05 showed that the same
work continued to a normal result in about 142 seconds. Therefore the earlier
partial artifacts are classified as `SUPERVISION_WINDOW_TOO_SHORT`, not
training, XLA, GPU, target, or tuner failures.

## Checks

- Focused suite: 9 tests passed.
- Registry: five executable, seven blocked inventory cells.
- Scoped compile and `git diff --check`: passed.
- Native tuner configuration: target `0.70`, band `[0.65,0.75]`, empty fixed
  grid fields, identity `z` mass.
- GPU memory at terminal inspection: approximately 2.6 GiB, with memory growth.

## Phase 3 handoff

Entry conditions are satisfied. Launch the single Python campaign command in a
supervised session with a fresh root. Each cell runs in its own child process.
Continue across valid scientific candidate rejection; stop on process failure,
missing result artifact, target drift, device-policy failure, or shared harness
invalidity. Do not launch the seven blocked inventory cells.

The run may take hours because each executable cell includes up to four fresh
500-step screens, one fresh 5,000-step final training, native tuning, and
adaptive warm-up/retained sampling. Short periods without terminal output are
not failures; use structured progress and process/GPU state.

Forbidden claims remain: no conclusion from the preflight about HMC validity,
truth recovery, convergence, method superiority, or default readiness.
