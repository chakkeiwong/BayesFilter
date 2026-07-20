# SSL-LSTM q=20 Wide NeuTra Capacity Saturation Test

Date: 2026-07-20  
Tier: 2 material GPU/XLA research engineering  
Status: `AUDITED_FOR_EXECUTION`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Does widening each IAF hidden block from `(32,32)` to `(64,64)` delay or avoid the q=20 scale-saturation veto at batch size 100? |
| Exact baseline | Existing q=20 batch-100 seed-a diagnostic with tuned three-stage `(32,32)` dense IAF, fixed-smoke optimizer parameters, 250-step validation, 2,000-step cap, and saturation cap `0.05`. |
| Candidate mechanism | Same target, three IAF stages, ELU, reverse-coordinate mixing, fixed translation, bounded `s_max=1`, adaptive training, seed, and batch; only hidden widths change to `(64,64)`. |
| Primary diagnostic | First validation step with aggregate `saturation_fraction > 0.05`; stage-level saturation and parameter count explain the mechanism. |
| Promotion criterion | None. This is a one-seed capacity diagnostic and cannot admit HMC or establish posterior correctness. |
| Hard vetoes | Nonfinite values, target/chart/signature mismatch, invalid serialization/reload, round-trip failure, host-memory breach, GPU/XLA/memory-growth failure, or missing telemetry. |
| Continuation veto | Any changed setting prevents isolation of width, or the artifact is incomplete/corrupt. |
| Repair trigger | Code/test failure receives a focused repair; a saturation veto is candidate evidence and does not reject NeuTra or q=20. |
| Explanatory diagnostics | Loss, gradients, clipping, wall time, parameter count, stage saturation, and memory. |
| Nonclaims | No optimal-width, superiority, transport-quality, HMC-convergence, predictive-equivalence, posterior-validity, or default-readiness claim. |

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| `(64,64)` | User-requested width hypothesis | Isolates width from the preceding `(32,32,32)` depth test. | Extra parameters may increase scale growth or runtime; record parameter count and stage telemetry. |
| q=20, batch 100, seed-a | Matched prior diagnostic | Keeps the comparison paired with the existing baseline. | One seed is descriptive only. |
| Fixed-smoke optimizer and controller | Inherited baseline | Avoids confounding architecture with learning-rate or stopping changes. | Width may require retuning; this run cannot establish tuned performance. |
| GPU/XLA float64 parent and CPU-hidden workers | Existing execution route | Preserves numerical and process topology comparability. | Resource failures are not scientific evidence. |

## Evidence Contract

Baseline artifact:
`docs/plans/artifacts/ssl-lstm-batch100-q20-pipeline-final-2026-07-20/training/seed-a/result.json`.
Candidate artifact root:
`docs/plans/artifacts/ssl-lstm-q20-wide-64x64-saturation-2026-07-20/run-01/`.

The candidate manifest must preserve the exact command, commit, dirty state,
device/JIT/TF32 settings, seed, tuple, parameter count, validation history,
stage saturation, stop reason, and artifact hashes. HMC is not part of this
run and must remain withheld.

## Skeptical Pre-Execution Audit

- Wrong baseline: passed; q, seed, batch, optimizer, cadence, cap, and target
  are matched to the prior `(32,32)` diagnostic.
- Proxy promotion: passed; saturation timing is descriptive and cannot admit
  HMC or establish scientific validity.
- Missing stop: passed; 250-step checks, 2,000-step maximum, hard vetoes,
  resource cap, and resumable checkpointing remain active.
- Unfair comparison: passed; the only intended mechanism change is width.
- Hidden assumptions: recorded; one seed and inherited optimizer settings are
  explicitly unpromoted hypotheses.
- Artifact adequacy: passed; runner and loader bind the wide procedure label
  and hidden tuple in manifests and frozen payloads.

Audit decision: `PASS_FOR_ONE_BOUNDED_Q20_WIDTH_DIAGNOSTIC`.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
  --mode single-diagnostic --q 20 --batch-size 100 \
  --hidden-layers 64,64 --authorize-material-run \
  --gpu-cap-seconds 7200 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-q20-wide-64x64-saturation-2026-07-20/run-01
```

## Interpretation Rules

- A later saturation step is evidence only that this width changes this
  failure mode under one seed and fixed-smoke optimization.
- Equal or earlier saturation means width alone did not rescue the candidate;
  investigate scale parameterization and optimizer interaction next.
- No saturation through 2,000 steps still does not establish convergence or
  HMC readiness.
- Any hard numerical, artifact, target, or launch veto invalidates the run as
  evidence rather than ranking the candidate.
