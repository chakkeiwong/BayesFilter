# SSL-LSTM q=20 Deep NeuTra Capacity Saturation Test

Date: 2026-07-20  
Tier: 2 material GPU/XLA research engineering  
Status: `AUDITED_FOR_EXECUTION`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Does adding a third width-32 hidden layer to the three-stage SSL-LSTM dense IAF delay or avoid the scale-saturation veto observed for the q=20 `(32,32)` transport at batch size 100? |
| Exact baseline | Existing q=20 batch-100 fixed-smoke run using the tuned SSL-LSTM three-stage dense IAF with hidden layers `(32,32)`, the same fixed-smoke optimizer parameters, seed-a, validation every 250 steps, 2,000-step maximum, and saturation cap `0.05`. |
| Candidate mechanism | Same three IAF stages, ELU, bounded `s_max=1`, reverse-coordinate mixing, fixed translation, adaptive training, and target; only each IAF hidden-layer tuple changes to `(32,32,32)`. |
| Primary diagnostic | First validation step at which `saturation_fraction > 0.05`; if no veto occurs, terminal saturation fraction and maximum completed validation step. |
| Promotion criterion | None. A later saturation time is descriptive evidence that added capacity changes this failure mode only. It does not admit HMC or establish posterior correctness. |
| Hard vetoes | Nonfinite values, failed target/chart/signature checks, invalid artifact serialization/reload, round-trip failure, host-memory cap breach, GPU memory-growth/XLA launch failure, or missing required telemetry. |
| Continuation veto | The comparison cannot isolate hidden-layer depth because another setting changes, or the result artifact is incomplete/corrupt. |
| Repair trigger | A code/test failure triggers a focused repair; a saturation veto triggers interpretation as candidate failure and does not invalidate the architecture direction. |
| Explanatory diagnostics | Loss trajectory, gradient/clipping diagnostics, wall time, parameter count, per-stage saturation, and memory are explanatory only. |
| Nonclaims | No claim of better transport quality, posterior validity, HMC convergence, predictive equivalence, optimal architecture, superiority, or default readiness. |

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| `(32,32,32)` | User-requested capacity hypothesis | Tests depth while retaining width and all other topology choices. | More parameters can slow or destabilize training; record parameter count and wall time. |
| q=20, batch 100, seed-a | Existing batch-100 q=20 diagnostic comparator | The prior q=20 streams saturated early; one fixed seed gives a clean paired mechanism test. | One seed is not robust evidence; no multi-seed or HMC claim follows. |
| Fixed-smoke parameters | Inherited unpromoted hypothesis | Prevents optimizer changes from confounding capacity. | Saturation may be an optimizer/scale problem; this test cannot distinguish all causes. |
| 250-step checks, cap 0.05, max 2,000 | Existing controller policy | Preserves the current admission and stopping contract. | Sparse checks can miss transient behavior; full validation history is retained. |
| GPU/XLA, float64, batch-native external score pool | Repository execution default and existing comparator | Keeps the comparison on the same numerical route as the baseline. | Resource or allocator failure is a launch/resource result, not scientific evidence. |

## Evidence Contract

The exact baseline artifact is
`docs/plans/artifacts/ssl-lstm-batch100-q20-pipeline-final-2026-07-20/`.
The candidate artifact is written under
`docs/plans/artifacts/ssl-lstm-q20-deep-32x32x32-saturation-2026-07-20/`.

The candidate must record the exact command, git commit and dirty state,
TensorFlow/XLA/TF32/device settings, seed, hidden layers, parameter count,
batch size, validation history, saturation telemetry, stop reason, and output
hashes. The run is valid only if all finite/support/round-trip checks pass and
the candidate topology is visible in both the manifest and frozen payload.

## Skeptical Pre-Execution Audit

- Wrong baseline: passed. The comparator is the completed q=20 batch-100
  `(32,32)` run, not a different q, seed, optimizer, or saturation policy.
- Proxy promotion: passed. Saturation timing is the primary diagnostic only;
  it cannot admit HMC or establish scientific correctness.
- Missing stop: passed. The existing 2,000-step cap, 250-step checks, hard
  vetoes, and host/GPU resource caps remain active.
- Unfair comparison: passed. The only intended change is hidden-layer depth;
  the candidate receives a distinct family and artifact procedure label.
- Hidden assumptions: recorded above. Parameter count and per-stage telemetry
  are required to expose capacity/resource effects.
- Artifact adequacy: passed. The runner writes resumable progress, result,
  manifest, frozen transport, and source bindings.

Audit decision: `PASS_FOR_ONE_BOUNDED_Q20_DEPTH_DIAGNOSTIC`.

## Implementation Scope

1. Add an explicitly named SSL-LSTM deep-capacity family and factory for
   `(32,32,32)`; preserve the existing `(32,32)` family and loader contract.
2. Extend the complexity-training runner with an explicit `--hidden-layers`
   option, defaulting to `(32,32)`, and bind it into manifests and trainer
   construction. The diagnostic command will pass `32,32,32`.
3. Extend artifact validation for the deep procedure label and expected
   hidden-layer tuple.
4. Add focused contract tests and run them CPU-hidden before the GPU run.
5. Run one bounded q=20 seed-a GPU/XLA diagnostic with batch size 100 and the
   existing fixed-smoke parameters. Do not launch HMC.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
  --mode single-diagnostic --q 20 --batch-size 100 \
  --hidden-layers 32,32,32 --authorize-material-run \
  --gpu-cap-seconds 7200 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-q20-deep-32x32x32-saturation-2026-07-20/run-01
```

## Stop And Interpretation Rules

- Stop immediately on any hard launch, numerical, target, serialization,
  support, round-trip, or memory veto.
- If saturation occurs later than the `(32,32)` baseline, report the delay as
  descriptive evidence for this failure mode only.
- If saturation does not occur within 2,000 steps, report that the candidate
  avoided this screen under one seed; do not call it converged or admitted.
- If saturation occurs at the same or earlier step, the extra depth did not
  rescue this candidate under the fixed-smoke settings; assess optimizer and
  scale parameterization next.

## Planned Result Record

Create
`docs/plans/bayesfilter-ssl-lstm-q20-deep-32x32x32-saturation-test-result-2026-07-20.md`
with the command actually run, artifact paths, validation table, decision and
inference-status table. State explicitly whether the result invalidated the
harness, implementation, target, data, math, or only the candidate setting.
