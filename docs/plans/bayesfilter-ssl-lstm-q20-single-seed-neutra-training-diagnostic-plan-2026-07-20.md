# SSL-LSTM q=20 Single-Seed NeuTra Training Diagnostic Plan

Date: 2026-07-20  
Tier: 2 material GPU/XLA research engineering  
Status: `COMPLETED_DIAGNOSTIC_RUN`

## Research Intent And Evidence Contract

| Role | Contract |
| --- | --- |
| Question | Does one complete q=20 NeuTra stream execute the revised 2,000-step/250-step adaptive mechanism correctly, including CPU value/score workers, GPU/XLA transport updates, heldout validation, best-state restoration, 50% LR repair, plateau stopping, checkpoint resume state, transport freezing, and support probes? |
| Exact target | `complexity_posterior_target(q=20, jit_compile=True)`, principal-square-root UKF, four controlled free coordinates, 30 observations. |
| Candidate | One fresh execution of fixed stream `seed-a`, batch 480, validation batch 64, 32x32 three-stage dense IAF, maximum 2,000 steps, validation/patience 250, two post-repair no-improvement cycles. |
| Hyperparameters | Diagnostic warm-start only: learning rate `4e-4`, initialization scale `0.01`, clip norm `10`. These are the runner's existing fixed-smoke parameters; no q=20 Optuna nomination exists. |
| Primary mechanism criterion | The stream reaches a declared controller terminal state or a valid resumable resource stop; every completed validation/checkpoint is finite, internally bound, and replayable; any LR repair occurs only after best trainer/Adam restoration. |
| Per-seed transport screen | Report the existing heldout-improvement, saturation, round-trip, and moderate-shell support checks as a diagnostic stream gate. Passing it is not Phase 3 admission. |
| Hard vetoes | Nonfinite training/validation, saturation above 0.05, support or round-trip invalidity, checkpoint inconsistency, worker-visible GPU, wrong worker count/order, host estimate above 64 GiB, GPU memory-growth failure, source/config drift, or corrupt/missing artifacts. |
| Resource stop | Eight cumulative GPU-hours, checked before each optimizer step with a 60-second reserve; write a joint trainer/controller/best-state checkpoint before stopping. |
| Explanatory diagnostics | Loss history, paired heldout delta/UCB, best and terminal step, LR history/actions, gradient norms, wall time, allocator/RSS evidence, and support radii. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/`. |
| Nonclaims | No q=20 hyperparameter nomination, seed robustness, Phase 3 admission, NeuTra adequacy, HMC readiness/convergence, posterior correctness, superiority, default readiness, or scientific validity. |

## Budget Derivation

The current q=20 process-parallel receipt measured maximum warm optimizer-step
time `12.058894310845062` seconds with 16 CPU workers and batch 480. The
2,000-step upper bound projects `24,117.79` seconds (6.70 hours) before startup,
eight validation/support boundaries, freezing, and reporting. The diagnostic
cap is 28,800 seconds (8 hours), an approximately 19.4% envelope above the
warm-step projection. This is a measured-rate resource bound, not a claim that
the run will need or complete all 2,000 steps. Adaptive plateau stopping may
return substantial time.

## Default And Assumption Audit

| Choice | Provenance/status | Why used | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| q=20 and one seed | Owner-requested diagnostic | Tests the largest current state-complexity rung and full adaptive mechanism at minimum replication cost. | One stream can be atypical; the artifact is barred from Phase 3 admission and ranking. |
| Fixed-smoke hyperparameters | Existing convenience baseline, unpromoted hypothesis | No q=20 nomination exists; these parameters allow a mechanism run without pretending cross-q tuning evidence. | Poor tuning can veto the stream for the wrong reason; verdict separates mechanism execution from transport adequacy and triggers q=20 tuning rather than idea rejection. |
| 2,000/250/two-cycle schedule | Owner-reviewed prospective protocol | This run directly exercises the newly implemented policy. | It may truncate continuing improvement or use sparse checks; terminal history is preserved and no optimality claim follows. |
| Batch 480, validation 64 | Inherited q-general baseline | Preserves the runner contract and measured timing topology. | Validation power may be weak and compute costly; UCB and worker timing are recorded. |
| 16 CPU workers, one GPU | Measured q=20 selected topology | Prior receipt found 32 workers only 0.3% faster with much higher RAM. | Concurrent lanes can contaminate rate; preflight checks GPU and process activity, while performance remains explanatory. |
| Eight-hour cap | Derived from measured 2,000-step rate plus 19.4% envelope | Bounded opportunity to complete the requested stream. | Unexpected slowdown may cause a valid resource stop; checkpoint resume remains possible under new authority. |

## Premortem

- The command could succeed while the diagnostic is scientifically misleading
  if a single seed is called robust or its convenience hyperparameters are
  called tuned. The output schema and result language forbid both claims.
- It could fail because the fixed-smoke LR is poor rather than because NeuTra
  cannot model q=20. The result must classify this as a candidate/tuning
  failure unless implementation or target validity fails.
- It could appear to plateau because the 64-row heldout UCB has low power. The
  per-sample paired interval and exact history remain available; the run does
  not establish optimal stopping.
- It could consume excessive RAM if workers duplicate more state than the
  prior canary. The existing 64 GiB aggregate high-water veto remains active.
- It could reserve most VRAM despite low tensor use. TensorFlow memory growth
  is set and verified before logical-device initialization; failure is a launch
  veto.

## Skeptical Pre-Execution Audit

- Wrong baseline: passed with qualification. The hyperparameters are explicitly
  an unpromoted diagnostic baseline, not a q=20 optimum.
- Proxy promotion: passed. Heldout loss controls checkpoints and a per-seed
  screen only; it cannot admit q=20 or validate a posterior.
- Missing stop: passed. Plateau, maximum-step, hard-veto, and eight-hour
  resource stops are explicit and resumable where valid.
- Unfair comparison: not applicable; no method or candidate ranking is made.
- Hidden assumptions: q, seed, architecture, parameters, batch sizes, cadence,
  topology, precision, JIT, and budget are declared above.
- Environment mismatch: trusted GPU1 is currently free; workers remain
  CPU-hidden. The run must re-probe at launch.
- Artifact adequacy: the existing progress/checkpoint/result machinery records
  every validation boundary and binds trainer, Adam, controller, and best state.

Audit decision: `PASS_FOR_ONE_BOUNDED_Q20_SINGLE_SEED_DIAGNOSTIC`.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
--mode single-diagnostic --q 20 --authorize-material-run \
--gpu-cap-seconds 28800 \
--params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
--output-root docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/run-01
```

The first launch was preserved as an invalid launch receipt: its late
memory-growth call failed after project imports had initialized TensorFlow. The
second fresh launch reached the pool but failed closed because spawned workers
re-imported the benchmark and overwrote their inherited CPU-hidden environment
with the parent GPU selection. Neither attempt performed a training step or
contributes scientific evidence. The runner was repaired and the
process-parallel visibility suite rerun before launching `run-03`.

## Close Record

The valid execution is `run-03`. Its result and summary are recorded at:

- `docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/run-03/seed-a/result.json`;
- `docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/run-03/single-diagnostic-summary.json`.

The stream reached step 750 and stopped with
`scale_saturation_above_cap`. Steps 250 and 500 produced support-eligible
meaningful improvements; at step 750 the saturation fraction was
`0.059895833333333336`, above the hard `0.05` cap. Therefore the live run did
not exercise learning-rate halving: saturation is an independent continuation
veto evaluated before plateau repair. The controller's `+250 repair`,
two-post-repair-cycle, and resume semantics remain covered by focused tests,
but this q=20 stream cannot provide live evidence for that branch.

Run result: `DIAGNOSTIC_VETOED`; underlying per-seed gate: `VETOED` solely for
`dense_scale_saturation_above_cap`. This is a fixed-smoke hyperparameter
candidate failure, not a target-invalidity, worker-boundary, memory-cap, HMC,
posterior, or NeuTra-direction rejection.
