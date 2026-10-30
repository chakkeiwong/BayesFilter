# SIR NeuTra-only continuation

The owner requested one working method and removed the dense-affine comparison.
The comparator and its supervisor were stopped; 6.072536 HMC hours remain
charged to that attempt. Including completed mechanics diagnostics, 17.674803
hours of the original 24-hour HMC allowance remain. Training and frozen-map
checks are reused from the preceding roots. The sampler, scientific prior,
fixed MH endpoint, trained map and diagnostic thresholds are unchanged.

The master now launches only frozen NeuTra and assesses its retained sampling
without requiring a comparator. Twenty-four focused CPU tests pass, including
fresh/resumed frozen-map binding, scheduling, charged-budget preservation and
failure on missing or inadequate diagnostics. CPU checks intentionally hide the
GPU. Evidence and the source snapshot are in `preparation/`.

Launched at 2026-10-22 18:25:35 Hong Kong; by 18:26 the actual configuration
records `sampling_coordinates=frozen_neutra_z`. Worker PID 3542305 initialized
GPU/XLA on RTX 5080 with verified memory growth and TF32 disabled. It reuses
`../sir-neutra-training-20261022-repair01/016-freeze/result.json`; all earlier
prerequisite stages were cached. Exact launch command:

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_sir_neutra_training.py run --resume-from docs/plans/artifacts/sir-neutra-training-20261022-repair01 --repair-note docs/plans/sir-neutra-training-execution-2026-10-21.md
```

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Continue NeuTra only | Retained sampling R-hat <=1.01, bulk/tail ESS >=400, MCSE/SD <=.10 | Training/map checks passed; posterior assessment pending | Whether SIR mixing is adequate within the remaining time | Tune, adapt warmup, retain cumulative samples | Convergence, exact-posterior accuracy or superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new failure; comparator stopped by owner direction |
| Statistically supported ranking | Not requested; no ranking claimed |
| Descriptive differences | No new stochastic comparison |
| Default readiness | Outside this narrowed objective |
| Next evidence needed | NeuTra cumulative posterior diagnostics |

The [execution plan](../../sir-neutra-training-execution-2026-10-21.md) records
the pre-run skeptical audit. `campaign.json` carries every prior attempt and
budget charge; `next-phase.json` records progress. Each new worker has a unique
directory, exact command, source hashes, hardware/memory policy and saved log.
Termination of the comparator is neither a failed NeuTra candidate nor a
rejection of the sampling method. Removing the comparison cannot establish
sampling quality: that remains the question for this continuation.
