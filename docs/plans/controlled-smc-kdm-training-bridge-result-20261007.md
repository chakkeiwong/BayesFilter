# KDM controller and native-consumer result — 7 October 2026

The bounded bridge phase passes: **423/423 source controller checks,
1648/1648 GPU native-consumer checks, and 9/9 focused guard/restore checks**.
Both complete replications remain unfinished. Plan:
[controller/native continuation](controlled-smc-kdm-training-bridge-20261007.md).
Machine result: [summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-training-bridge-20261007-summary.json).

The host controller executes the source's strict best-checkpoint comparison,
relative plateau scheduler, cooldown, delayed start, early-stop offset and
tie behavior. The oracle executes the unchanged AST bodies of the pinned
source EarlyStopping, CustomReduceLROnPlateau and Trainer.train, with plotting
and file saves observed through stand-ins. Source numerical training is not
substituted by these synthetic validation streams. The installed Torch
requires an inherited is_better/_is_better name alias and omission of the
removed logging-only verbose keyword; neither changes the decision formulas.

The neural consumer now draws its resampled particles and proposal innovations
inside its GPU/XLA recurrent computation. Captured draws replay through the
existing source-matched path with matching loss, gradients, clipping, Adam
moments and parameters for MDPF/A-MDPF in float64/float32. Learning rates are
live variables and source-controller reductions affect updates without
retracing. Checkpoints preserve all frozen/trainable parameters, Adam moments,
step, rates and constants; on-disk restore reproduces the next update exactly.
Two fresh consumers restore the same checkpoint and produce identical states.

A reject-only guard computes candidate Adam updates before committing any
state. The source replay is its no-fire regression. Invalid bandwidth and a
deliberately exhausted circular rejection loop both reject without changing
parameters, moments or step. The cap1 diagnostic has finite loss/gradients,
so it specifically checks sampler-status propagation. The normal cap remains
128. This does not stabilize or improve the author method.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | What is not concluded |
|---|---|---|---|---|---|
| Accept controller implementation screen | 423 exact event/state checks pass | No controller mismatch | Synthetic loss streams; original Torch runtime has API drift | Wire complete source training stages | No learning-quality claim |
| Accept native consumer wiring screen | 1648 checks pass, both dtypes/variants | Validity, parameter coverage and one-trace checks pass | Initial state is still the captured source fixture | Implement native initialization and complete epochs | No stochastic no-worse qualification |
| Accept checkpoint/reject-only behavior | 9 focused checks pass | Exhaustion/incomplete-checkpoint guards reject correctly | Epoch/data cursor and full stage resume not yet implemented | Add full training-driver state | No full training completion |

## Timing and execution
The source trial's train64 and validation256 shapes are covered. B256 uses
repeated fixture rows for a shape/cost check, not the validation dataset.
After compilation, sampled B64 update times ranged about 0.14–1.31 seconds;
B256 evaluation was about 0.77–0.78 seconds. Other GPU work was active, and
there are too few observations for a speed ranking or full-training estimate.
All checked functions trace once, use XLA, TF32 off and verified memory growth.
Native-run allocator peak was 274,727,168 bytes. Source-reference checks
explicitly hide CUDA.

Four versioned attempts preserve full logs and manifests. Attempt01 failed
only when serializing the source scheduler's initial infinity sentinel;
attempt02 encodes that uninitialized state as null and passes. Attempts03
and04 pass on GPU. No source algorithm or tolerance was changed.
Measured CPU launcher wall time is 4.739317s; a conservative 30s
allowance covers the preliminary unlogged CPU import/API probes.
Charge **34.739317 CPU seconds** and **261.898278 GPU seconds** once
to the existing allocation. CPU/GPU phase caps of 1800/3600s were respected.
Commands, code/source hashes, seeds, environments and output checksums are in
each attempt's manifest; no new compute allocation is assumed.

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | No new failure on these valid fixtures; deliberate invalid cases rejected |
| Statistically supported ranking | None; exact implementation diagnostics only |
| Descriptive-only differences | Runtime and allocator observations under shared GPU load |
| Default-readiness | Not established; complete source training and qualification still open |
| Next evidence | Native initialization/pretraining/full data epochs, then frozen source-only calibration and untouched comparisons |

The earlier float32 circular-gradient and B2 neural-angle failures remain open.
A passing new fixture does not erase them or widen the accepted bounds.

Post-run red-team: the strongest alternative explanation for apparent completion
would be mistaking checked local updates for replication of a trained method.
The initial cloud and data slice are fixtures, pretraining is absent, and no
full epoch or final evaluator is certified here. Those facts prevent that
conclusion. A fresh source case exposing a value/gradient or controller mismatch
would overturn the relevant implementation screen. The next step is
Implement source true-state initialization, weight pretraining, shared-parameter batch specializations and complete data epochs, with source stage/checkpoint transfer. Then measure an actual native epoch before source-only cost/power calibration and untouched paired qualification. Controller/native wiring passed; preserve prior float32 failures and the cSMC conditional-solver gap; no E0.
