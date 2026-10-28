# Native TFP transition and Stan comparison

Status: skeptical pre-execution review passed; implementation and bounded
comparison complete. See the [result](bayesfilter-stan-tfp-native-result-2026-10-28.md). Owner
request: rewrite the TFP-compatible wrapper to use native TFP and benchmark
against Stan. This supersedes the wrapper's direct-tree implementation, not
the preserved direct Stan port or its historical evidence.

## Question, method and evidence contract

Does replacing the direct-tree adapter with stock TFP NUTS remove avoidable
transition overhead while preserving the declared target and practical
sampling validity? Use tfp.mcmc.NoUTurnSampler, TFP transformed kernels for
fixed metrics, and the existing audited Stan-style adaptation around native
TFP. Keep target/gradient caches rather than rebuilding them every step.
Use tfp.mcmc.sample_chain inside a stable-signature XLA enclosing function.
No custom tree algorithm, scalar Python sampling loop or numerical fallback.

Baselines: pinned CmdStan 2.37.0; existing direct port as a separately priced
control on Gaussian32 and banana8. The main TFP/Stan comparison covers the
same seven targets (Gaussian32, rotated Gaussian32, banana8, centered and
noncentered funnel10, regression32/64). Reuse the existing target definitions,
native binary and common posterior diagnostics. This is a comparison of
native TFP with Stan-style adaptation to native Stan, not a source-identical
Stan transition; the difference must be explicit in API/docs/results.

Engineering pass: native-TFP common-seed equivalence, correct inverse-metric
orientation/caches, warm-up handoff, seeded reproducibility, exact chunk
continuation, invalid-state handling, and enclosing XLA compilation.
Performance primary quantity: reusable warm-up + retained numerical wall
time; report warm-up and retained costs separately and first-call compilation
+execution separately. Also report minimum bulk/tail ESS per elapsed second
as descriptive only. Native wall includes startup/CSV; TF serialization is
separate and must be added for a workflow-cost column. No unqualified claim
that the remaining boundaries match. Report native printed numerical times
as secondary, with their own meaning.

Promotion veto: failed reference moments, retained state/cache health, chain
movement, R-hat >1.01, bulk/tail ESS <400 or standardized MCSE >.05. Those
inherited screens qualify each result; acceptance, leapfrogs, divergences,
depth caps, compilation, memory and elapsed time explain results and never
establish convergence. Centered funnel failure is an expected candidate
failure; proceed to the noncentered control rather than abandoning the study.
Continuation veto: wrong target/gradient, broken RNG/metric/cache, missing
required artifacts, XLA fallback, budget exhaustion or unsafe resource usage.
Local API/harness failures trigger repair and fresh attempt directories.

Three fixed seeds (20261028/29/30) give descriptive medians, not statistical
superiority. No default promotion, broad GPU/CPU ranking, convergence beyond
the tested observables, fixed-HMC claim, or exact Stan RNG/tree reproduction.
Do not promise that native TFP guarantees speed: measure it.

## Assumptions/defaults and pre-mortem

- FP64, four independent chains, 1000 warm-up/2000 retained, delta=.8,
  windows75/50/25, depth10, jitter0: inherited comparison/Stan configuration.
  Failure risk: hard geometry remains underexplored. Early health/reference
  screens expose it; such cells cannot support efficiency promotion.
- CPU scalar chains, one thread each, at most four concurrent: earlier
  separately calibrated study selected this layout for these exact targets.
  Retain it for a controlled CPU comparison, not a claim that this new wrapper
  is the fastest possible GPU layout. CPU hides GPUs before import.
- Diagonal uses elementwise scale; dense uses ScaleMatvecTriL. With q=A z
  and A A^T=C (inverse mass), physical momentum A^-T p_z gives
  p_q^T C p_q=p_z^T p_z; qdot=C p_q=A p_z. TFP supplies the constant
  log-Jacobian. Verify transformed value/gradient against independent math.
- Initial positions, target data and seeds identical by fit across engines;
  same numeric seed is not common random variates between TF and Stan.
- Source TFP constructor builds static instruction metadata internally;
  repository numerical code remains TensorFlow/TFP with no added NumPy path.
- Separate compilation on disjoint seed20260901; reuse numerical functions
  without tracing on tuned values. Alternate engine order to reduce drift.
- Potential misleading success: speed without mixing, omitted tuning costs,
  testing an old path instead of new code, or moving metric transform into
  every leapfrog. Check actual dependency hashes, call route, diagnostics,
  metric algebra and measured phase costs before interpretation.

Audit verdict: proceed. Native TFP changes the transition baseline explicitly.
No old direct-tree parity claim carries over; prior adapter sources are
archived as text. Existing seven-target diagnostics and native density checks
avoid a weak convenient baseline. No module import uses the wrong Python
environment or unconfigured GPU; all workers set visibility/memory first.

## Execution and budget

1. Rewrite adapter and tests; document migration and TFP/Stan boundary.
2. Run focused CPU/XLA tests; validate native equivalence and health/cache
   behavior. Stop and repair numerical defects before comparison.
3. Run new actual adapter versus Stan, seven targets x three seeds, with
   a fresh direct-control on Gaussian32/banana8. Record full outputs and HLO.
4. Summarize posterior screens first, timings second; review and reset note.

Environment `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, TF2.20/TFP0.25.
CPU variables before import: CUDA_VISIBLE_DEVICES=-1,
TF_FORCE_GPU_ALLOW_GROWTH=true, BAYESFILTER_PRELOAD_CUSTOM_OP=0. GPU remains
the repository default outside this explicit controlled reference benchmark.
Artifacts: docs/benchmarks/artifacts/stan-tfp-native-20261028, one fresh
subdirectory per launch. Exact commands/source hashes/commit/environment/data/
seeds/wall and result paths go in launch and worker manifests.

At most eight subprocess attempts and 1200 total subprocess wall seconds
(convenience-chosen ceiling for small fixture compilation/comparison).
Individual native workers retain the existing 300-second watchdog; outer
launch limit is min(remaining campaign allowance, requested watchdog).
Repair/retry within the same question/budget needs no new approval. If this
budget cannot produce adequate evidence, report the measured subset and
remaining blocker; do not fabricate completion.

Result: docs/plans/bayesfilter-stan-tfp-native-result-2026-10-28.md.
