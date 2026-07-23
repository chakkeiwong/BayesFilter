# q=20 SSL-LSTM `(32,32)` NeuTra-HMC Tuning

Date: 2026-07-21  
Tier: 2 material GPU/XLA sampler tuning  
Status: `PREFLIGHT_PASSED_READY_FOR_FRESH_TUNING`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Do the two selected q=20 `(32,32)` loss-only NeuTra transports define valid transformed targets and admit separately confirmed fixed HMC kernels near acceptance 0.70? |
| Exact inputs | The `seed-a` and `seed-b` `ADMITTED` result receipts under `docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21/arch-32x32/`. |
| Candidate mechanism | Identity-mass HMC in each frozen NeuTra chart, batched four-chain TensorFlow/TFP XLA execution. |
| Primary tuning criterion | Fresh four-chain confirmation has every chain's mean Metropolis acceptance probability in `[0.60,0.80]`, finite telemetry, and movement in every chain. Realized binary acceptance is explanatory only. |
| Hard vetoes | Frozen-payload/hash/replay failure, GPU target-signature mismatch, transformed value/score or finite-difference failure, failed round trip, nonfinite HMC telemetry, unmoved chain, exposed positive divergence, host RSS above 64 GiB, GPU failure, or missing memory growth. |
| Explanatory diagnostics | Acceptance distance from 0.70, RMS jump, movement magnitude, runtime, allocator bytes, and scale/trajectory rows. These do not establish convergence. |
| Repair trigger | One geometric scale midpoint when the pilot band is strictly bracketed, and one adjacent trajectory confirmation for acceptance-only failure. |
| Continuation veto | Invalid transformed target, hard numerical/resource failure, or exhausted cap. A tuning miss rejects only the current kernel search. |
| Nonclaims | No posterior convergence, posterior correctness, retained-sample admission, predictive validity, architecture superiority, or default promotion. |

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| `(32,32)` | Owner engineering choice after an unresolved two-architecture gate | Both seeds are hard-valid and the smaller architecture is sufficient for this test | Not a statistically supported architecture ranking; retain that nonclaim |
| Two transports | Existing independent training seeds | Tests robustness to learned-chart initialization | Two seeds do not estimate broad training uncertainty |
| Target acceptance 0.70 | Existing reviewed HMC plan and user confirmation | Standard interior acceptance target for fixed HMC | Confirmation band, not point closeness, controls admission |
| Scale grid `0.05,0.10,0.20,0.40` | Existing bounded tuning runner | Coarse bracket before a single declared repair | May miss a viable scale; one midpoint/outer expansion is the only authorized repair |
| Trajectories `2,4,8,16` | Existing bounded tuning runner | Covers short through moderate trajectories | Identity mass may remain inadequate; report tuning failure rather than geometry success |
| CPU/GPU observation difference | Owner-authorized negligible roundoff | Maximum observed difference is `3.33e-16`; GPU HMC signature exactly matches frozen payloads | Record as ignored numerical roundoff; do not weaken payload-to-GPU signature binding |
| GPU memory growth | Owner directive and repository policy | Prevents TensorFlow from reserving nearly all VRAM | Launchers set and verify growth before project imports; absence is a launch veto |

## Skeptical Pre-Execution Audit

- Wrong baseline: checked; both selected receipts are `(32,32)`, q=20,
  loss-only, distinct-seed, `ADMITTED` artifacts.
- Proxy promotion: checked; training/audit loss selected the transports but does
  not promote HMC. Fresh HMC confirmation is required.
- Missing stop: checked; the runner has a cumulative cap, per-arm reserve,
  finite scale/trajectory grids, one midpoint repair, and one adjacent repair.
- Unfair comparison: checked; kernels are tuned separately and are not ranked.
- Environment mismatch: repaired; both timing and tuning launchers establish and
  verify TensorFlow memory growth before project imports.
- Stale budget: detected; refresh the q=20 current-source timing canary before
  setting the tuning cap.
- Artifact adequacy: checked; every arm binds the exact Phase 3 receipt, payload,
  target/transport signatures, seed, runner contract, and execution-source hash.
- Manifest provenance: repaired; the tuning runner now binds this live q=20
  `(32,32)` plan. The generic timing canary's already-recorded older complexity-
  ladder plan path is inherited metadata only; its command, source signature,
  target signature, measurements, and output path remain exact. The timing
  runner was not edited after measurement, so the rate artifact remains
  source-reproducible.

Audit decision: `PASS_AFTER_MEMORY_GROWTH_REPAIR_AND_RATE_REFRESH`.

## Execution Sequence

1. Run focused CPU-only contract tests and compile checks.
2. Run one q=20 current-source GPU/XLA timing canary using an untrained transport
   of the same `(32,32)` topology. This is cost evidence only.
3. Derive the tuning cap as `1.5 * refreshed_rate * 8,928 + 3,600 seconds`.
   `8,928` is the maximum transition-leapfrog workload across both transports;
   `3,600` covers at most six cold pilot/confirmation/repair runners. Round up
   to the next 600 seconds. This is a conservative ceiling, not expected time.
4. Run a separate transformed-target preflight with a 1,800-second cap.
5. If preflight passes, run the bounded tuning command under the derived cap.
6. Write a result note with per-arm kernels, confirmation diagnostics, resource
   use, hard-veto status, and nonclaims. Do not launch retained HMC.

## Commands And Artifacts

All GPU commands use `TF_FORCE_GPU_ALLOW_GROWTH=true`, GPU 1 when available,
TensorFlow/TFP, XLA, TF32, and trusted managed-session execution.

Artifacts are written under
`docs/plans/artifacts/ssl-lstm-q20-32x32-hmc-tuning-2026-07-21/`:

- `rate-refresh.json`;
- `preflight/summary.json`;
- `transformed-fd-ladder.json` and `preflight-repair-01/summary.json` when the
  original single-step finite-difference checker requires repair;
- `tuning/checkpoint.json`, `tuning/summary.json`, and per-arm receipts;
- a result note beside this plan.

The run stops on a hard veto or resource exhaustion. Unused cap is returned.
Before each non-preemptive arm, the checkpoint charges its prospective reserve;
after a successful durable arm receipt, it replaces that reserve with actual
cumulative time. Resume requires the exact input/source contract. Consequently,
an interruption can conservatively overcharge the cap but cannot erase prior GPU
cost or reuse arms under changed code.

## Preflight Checker Repair

The first preflight is preserved at `preflight/summary.json` with status
`PREFLIGHT_VETO`. Both charts were finite, exactly satisfied the implemented
change-of-variables value/score identities, replayed their payloads, and had
round-trip residual at most `2.10e-15`. The only veto was a central finite
difference at the convenience-chosen step `h=1e-5`: residuals were `5.39e-4`
and `5.71e-5` against a scale-aware `4e-5` tolerance.

The diagnostic ladder in `transformed-fd-ladder.json` (SHA-256
`a689ea0d52ff0c2760f86262b2bd43533356a203f5baf47455950410f309f4a2`)
shows the expected truncation/cancellation curve. Chart A error falls from
`2.21e-5` at `h=1e-2` to `1.65e-6` at `h=1e-3`, then rises to `5.39e-4` at
`h=1e-5`; chart B falls from `1.59e-5` to `7.15e-7`, then rises to `5.71e-5`.
Thus the failed single step is an ill-conditioned checker, not evidence of a
persistent score gap.

Prospective repair: use fixed steps `(1e-2,3e-3,1e-3)`, retain the existing
`atol + rtol * scale = 4e-5` ceiling at unit scale, and additionally require
the best error to occur after the first step and be at most half the first-step
error. The steps are target-calibrated from the diagnostic and are not a
universal default. The failed artifact remains immutable. A fresh repaired
preflight under the new source signature is required before tuning.

## Tuning Criterion Repair

The first source-bound tuning attempt is preserved at `tuning/summary.json`
(SHA-256
`61f0170aba1e668ebd41b8d4603495eab79fbbbb9ed61112a6e1a378880bb9cc`)
with status `TUNING_REPAIR_REQUIRED`. It charged `1,947.0304215629585`
seconds, had no preflight, hard, memory, or resource veto, and did not run a
confirmation. Both charts moved in every chain with finite telemetry through
the initial four scales. The runner nevertheless used the realized 16-draw
binary acceptance fraction as its scale gate. That is wrong relative to target
acceptance 0.70: the tuning quantity is the mean Metropolis acceptance
probability `mean(min(1,exp(log_accept_ratio)))`; binary acceptance is a noisy
movement diagnostic, as the shared HMC telemetry already declares.

The same run also showed that requiring every chain at every initial scale to
be above the pilot band before outer expansion is overrestrictive. Chart A's
mean acceptance probabilities at scale 0.40 were
`[0.9509,0.8513,0.9722,0.9623]`; chart B's were
`[0.9564,0.9676,0.9682,0.9326]`. These are directional evidence that a larger
scale is needed, but neither row passes the every-chain pilot band.

Prospective repair:

- use per-transition Metropolis acceptance probability, not binary acceptance,
  for pilot selection and confirmation;
- retain every-chain pilot `[0.50,0.90]` and confirmation `[0.60,0.80]` gates;
- use only the pooled boundary mean to choose whether to extend the scale grid;
- recognize a midpoint bracket when the lower scale reaches/exceeds the band
  without falling below it and the upper scale reaches/crosses the band without
  remaining above it, provided neither row has a hard veto;
- stop the outer expansion as soon as a viable scale or midpoint bracket is
  available.

This changes no target, transport, target acceptance, admission band, HMC
kernel, or retained-sample rule. The first attempt's charge is not reset. The
fresh `tuning-repair-01` cap is
`37,200 - 1,947.0304215629585 = 35,252.96957843704 seconds`; together the two
tuning attempts cannot exceed the original `37,200`-second authorization.

## Rate Refresh And Frozen Cap

The current-source q=20 timing canary passed under GPU/XLA with TensorFlow
memory growth verified. Its maximum warm rate was
`2.4970078943297267 seconds/transition-leapfrog`; first compiled call was
`148.99874091893435 seconds`; allocator peak was `381,568,512` bytes. The
artifact is
`docs/plans/artifacts/ssl-lstm-q20-32x32-hmc-tuning-2026-07-21/rate-refresh.json`
with SHA-256
`d8981f5a3385dbf9d906d5ce3385cac137a906070f1d164ebf996825cc6af1a2`.

The prospective calculation is

`1.5 * 2.4970078943297267 * 8,928 + 3,600 = 37,039.9297208637 seconds`.

Rounded upward to the declared 600-second boundary, the cumulative tuning cap
is `37,200 seconds` (`10.3333 GPU-hours`). This cap covers both transports and
all conditional repairs; it does not authorize retained HMC and does not imply
that all arms will run.

## Post-Merge Source Refresh (2026-07-22)

Remote `origin/main` was merged through `9303ed7` without conflict. The
pre-merge q=20 tuning attempts are source-stale and are not resumed. A fresh
current-source timing canary was written to:

`docs/plans/artifacts/ssl-lstm-q20-32x32-hmc-tuning-2026-07-21/rate-refresh-9303ed7.json`

The canary passed on GPU 1 with TensorFlow memory growth verified before
initialization, TF32 and XLA enabled, finite/moving four-chain mechanics, and
peak TensorFlow allocator memory `381,568,512` bytes. Its current execution
source signature is
`73603561bbfc1035203ce8a821b7d2dbc4a18d5931ebe74ce2398c730cf8d503`, and its
maximum warm rate is `1.463525768990318` seconds per transition-leapfrog. The
receipt SHA-256 is
`32dde90fe96261a94f81c7d59757bb1387ea2f1cf4faaf222b56a5f1bf9f6921`.

Using the q=20 plan workload formula, the fresh tuning ceiling is

`ceil_600(1.5 * 1.463525768990318 * 8,928 + 3,600) = 23,400 seconds`

(`6.5 GPU-hours`). This is a fresh tuning-attempt cap; it does not authorize
retained HMC and does not reuse the old source-stale charges as current-source
evidence.

A separate fresh transformed-target preflight passed under the merged source at
`docs/plans/artifacts/ssl-lstm-q20-32x32-hmc-tuning-2026-07-21/preflight-9303ed7/`.
It charged `283.35637893294916` seconds, had no chart vetoes, and used source
signature
`c80feb657e45673eaeada946822f670210d0b12bfdc85d0e69abbadefc783a44`. Its
summary SHA-256 is
`21a8a0c48ec8a557aebc37cccfa99f705fa3353f96b6a1530b2d82fe8937dff0`.
The preflight is an engineering-validity gate only; it is not HMC convergence,
posterior correctness, or transport-quality evidence.

The next action is a fresh bounded tuning run under `23,400` seconds and a new
versioned output root. Retained HMC remains forbidden until tuning and a fresh
fixed-kernel verification gate pass.

## Duplicate-Launch Incident And Lock Repair (2026-07-22)

The first current-source tuning root, `tuning-9303ed7/`, is invalid for
cumulative-budget accounting. A mistaken `--resume` launch (PID `4128368`)
ran concurrently with the original process (PID `4096559`) against the same
root. The duplicate was terminated before it wrote a distinct arm receipt, but
the shared checkpoint later regressed from `6499.714991202811` seconds to
`4977.73101590015` seconds. A cumulative charged value cannot decrease, so the
root is retained as debugging/resource evidence only. Its arm diagnostics are
not tuning evidence and no retained HMC may use them.

The runner now takes an exclusive advisory lock at
`<output-root>/.material-run.lock` for every material `preflight` or `tune`
execution. A second writer fails closed before loading targets or changing the
checkpoint. Focused regression coverage is in
`tests/test_ssl_lstm_neutra_complexity_hmc_tuning.py` and passed (`17 passed`).
Because the source signature changed, the prior `preflight-9303ed7/` receipt is
not reusable. The required repair is a fresh current-source preflight under a
new output root, followed by a fresh tuning root only if that preflight passes.

## Fresh Locked Run Result (2026-07-22)

The repaired preflight passed under
`preflight-lock-20260722/summary.json` (SHA-256
`582a3a966682321ca6fb466652db42e65e5d795576ff812326a7aeb4cc66bcdd`). The
single-writer tuning run under
`tuning-lock-20260722/summary.json` (SHA-256
`1e999b5181e60e74abd6a224832bb0bbbeca3048b6713aeddd00b27b6dcbabd6`) charged
`5040.527063304093` of the `23400` second cap and ended
`TUNING_REPAIR_REQUIRED`, with no resource or hard veto.

Chart A's midpoint scale and `L=2,4` pilots were viable, but its 64-draw
confirmation had per-chain acceptance
`[0.7983413071, 0.7061628912, 0.8480730187, 0.8043901365]`, so it failed the
`[0.60,0.80]` every-chain confirmation gate. Chart B's midpoint scale had
finite telemetry and pooled acceptance `0.7504800200`, but per-chain values
included `0.3982664516` and `0.9074824904`, so it failed the every-chain pilot
gate and no trajectory ladder was admitted. The detailed decision and
inference-status tables are in
`docs/plans/bayesfilter-ssl-lstm-q20-32x32-hmc-tuning-result-2026-07-22.md`.

Interpretation: this is a tuning-candidate failure, not a transformed-target
validity failure and not evidence against NeuTra geometry. Retained HMC remains
forbidden. The next phase requires a bounded, predeclared repair for per-chain
scale/confirmation robustness; it must not silently widen the admission bands
or promote pooled acceptance over the declared every-chain criterion.
