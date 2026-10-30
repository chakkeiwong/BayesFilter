# SIR NeuTra execution record

Started 2026-10-21 13:37:26 UTC on branch
`ledh-graceful-failure-execution-20260917`, baseline commit `591b36ee6` with the
source changes recorded by SHA-256 in `campaign.json` and worker manifests.
Plan: `docs/plans/sir-neutra-training-execution-2026-10-21.md`.

The question is whether a trained canonical IAF makes the existing SIR
surrogate-HMC sampler useful. The target is the existing T50 dataset and
scientific prior. Training uses the finite LEDH likelihood; MH uses the
independent fixed-design endpoint. Both arms use that same MH endpoint.

The authorized budgets are 16 GPU hours for prerequisites/training and 24
additional GPU hours for HMC. Compilation, diagnostics and failed attempts
count. Live consumption is recorded in `campaign.json`; `next-phase.json`
records the current action. The exact launch command was:

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_sir_neutra_training.py run
```

Engineering checks: 71 distinct focused CPU tests pass in their latest runs.
The GPU/XLA probe passed on the RTX 5080, with memory growth verified before
initialization and TF32 disabled. See `preparation/test-summary.json` and
`preparation/gpu-probe.json`. These checks do not establish learned-map quality.
Joint FP32 tuning for native B4 training and B2 HMC completed in 216.3 seconds.
Both validation datasets and both batches completed healthy trajectories.
The selected defensive mixture weights are approximately
`[0.1000000, 0.0560541, 0.8439460]`, with 16 flow steps, epsilon 1 and five
Sinkhorn iterations. The warmed native B4 value/score call took 5.18 seconds;
this is a descriptive timing, not a replicated speed comparison. Allocator
peak was 1,275,967,744 bytes across this worker's calibration and reference runs.

The maximum absolute FP32/FP64 log-likelihood difference was 0.000402; the
maximum absolute score difference was 0.1682. One score component's scaled
difference was 1.207%, so the old 1% screen reports false. This is an explanatory
precision warning, not a veto under the reviewed plan. The reference is the
same algorithm in FP64, not an exact SIR oracle. Exact batch-specific selections
and complete numerical evidence are in `001-tuning/`.

Preflight completed in 478.2 seconds. Scale 1 gave 27/32 valid initialization
points; scales .35, .1 and .02 each gave 32/32. The actual canonical map also
passed at .35, which was selected. All nine coordinate/location derivative
probes were finite at all three step sizes and showed no persistent mismatch
under the declared quantization-aware diagnostic. Some estimates vary with
step size, so this is not proof of exact score accuracy. The first update and
an independent subsequent batch were valid for clipping norms 1, 10 and no
clipping. Their subsequent losses were 2080.2402, 2080.2412 and 2080.2424;
these one-batch differences cannot rank the clipping policies. The raw initial
gradient norm was 514.7; subsequent clipping frequency remains a diagnostic.

The master advanced automatically to `003-w3-lr0.001-s100-r64/`. The first two
GPU/XLA optimizer updates were valid, with the second taking 9.13 seconds.
This is the first of six width/rate configurations, not a completed trained
map. Freeze checks and the adaptive HMC comparison remain planned automatic
stages. The unchanged source files used at launch are also preserved in
`preparation/source-snapshot.tar.gz`.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Conclusion not established |
|---|---|---|---|---|---|
| Continue reviewed campaign | Joint batch scope tuning and preflight passed | Broad initialization failed at scale 1; repaired at .35; no continuation veto | Learning, coverage and posterior mixing | Execute training ladder, freeze checks and adaptive HMC | Trained-map quality or posterior convergence |

| Inference status | Current evidence |
|---|---|
| Hard veto screen | Engineering, joint scope and preflight checks pass at selected initialization |
| Statistically supported ranking | None |
| Descriptive-only differences | Same-algorithm precision differences and one warmed timing above |
| Default readiness | Not established |
| Next evidence needed | Training/heldout results and adaptive same-endpoint HMC |

The principal alternative explanation for any later training success is a map
that fits the LEDH density locally while failing to cover the MH endpoint.
Fresh heldout positions and the downstream HMC comparison test that limitation.
This is an ongoing execution record; the terminal interpretation remains open.
