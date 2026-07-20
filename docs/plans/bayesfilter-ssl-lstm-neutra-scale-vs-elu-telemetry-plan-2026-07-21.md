# SSL-LSTM NeuTra Scale-vs-ELU Saturation Telemetry

Date: 2026-07-21  
Tier: 2 material GPU/XLA research engineering  
Status: `AUDITED_FOR_EXECUTION`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Are the q=20 SSL-LSTM saturation vetoes caused by hidden ELU preactivation tails, by the bounded IAF scale head, or by both? |
| Exact baseline | Existing q=20 seed-a `(32,32)` three-stage tuned SSL-LSTM dense IAF, batch size 100, fixed-smoke parameters, GPU/XLA parent, CPU-hidden q=20 score pool, validation every 250 steps, and the existing aggregate saturation cap `0.05`. |
| Candidate mechanism | Add compiled read-only telemetry only: raw scale logits before `tanh`, bounded scale logs, and hidden-layer preactivations for every IAF stage. No optimizer, activation, scale bound, target, or stopping-policy change. |
| Primary diagnostic | At each validation checkpoint, report per-stage scale-head saturation and raw-logit tails, plus per-stage hidden preactivation tails. |
| Promotion criterion | None. Telemetry is explanatory and cannot admit HMC or establish transport/posterior correctness. |
| Hard vetoes | Nonfinite telemetry, shape/order mismatch, changed forward/logdet behavior, target/chart/signature drift, serialization/reload failure, GPU/XLA/memory-growth failure, or missing required stage telemetry. |
| Continuation veto | The added telemetry changes the training path or cannot be aligned to the exact validation rows and IAF stage order. |
| Repair trigger | A focused implementation/test failure triggers a same-file repair; a training saturation veto remains candidate evidence and does not stop the diagnostic interpretation. |
| Explanatory only | ELU preactivation fractions, scale-logit fractions, loss, runtime, clipping, memory, and terminal saturation. |
| Nonclaims | No claim that ELU is optimal or faulty, no architecture ranking, no scale-bound change, no HMC readiness, no posterior correctness, and no predictive validity. |

## Mathematical Telemetry Definitions

For each IAF stage and validation row, let (a^{(\ell)}) be the hidden-layer
preactivation and (r) the raw scale output. The implementation computes

\[
s = s_{\max}\tanh(r/s_{\max}),
\qquad
u_i=z_i\exp(s_i)+t_i.
\]

The existing saturation screen is

\[
|s_i|\ge 0.95s_{\max}.
\]

The corresponding raw-logit threshold is

\[
|r_i|\ge s_{\max}\operatorname{atanh}(0.95)
\approx 1.83178s_{\max}.
\]

For ELU diagnostics, report fractions of hidden preactivations satisfying

\[
|a^{(\ell)}|\ge 5,
\qquad a^{(\ell)}\le -5,
\qquad a^{(\ell)}\ge 5.
\]

The value 5 is a diagnostic threshold, not a promotion criterion: on the
negative branch ELU has derivative (e^a\), so (a=-5) corresponds to a
derivative of approximately (0.0067). The telemetry must preserve the raw
values' finite min/max as well as these fractions.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| q=20, `(32,32)`, seed-a, batch 100 | Matched prior baseline | The prior three architecture tests used this exact comparator. | One seed is descriptive; no robustness claim. |
| Raw-logit threshold `atanh(0.95)` | Derived from the implemented scale equation | It distinguishes output-bound approach from hidden activation behavior. | A threshold is explanatory only; raw min/max remain available. |
| ELU preactivation threshold 5 | Reviewed diagnostic convention derived from ELU derivative | Identifies strongly small negative derivatives and large positive activations without changing training. | It is not a universal saturation constant; no hard gate uses it. |
| Read-only compiled telemetry | Implementation extension | Avoids changing gradients or optimizer semantics. | Extra compiled outputs may expose shape/XLA errors; focused tests and finite checks cover this. |

## Skeptical Pre-Execution Audit

- Wrong baseline: passed. The target, seeds, batch, parameters, controller,
  and cap remain unchanged.
- Proxy promotion: passed. All new metrics are explanatory only; the existing
  scale saturation veto is unchanged.
- Missing stop: passed. The existing 2,000-step, 250-step, resource, memory,
  and numerical stops remain active.
- Unfair comparison: passed. No activation or architecture is changed.
- Hidden assumptions: recorded above; thresholds are diagnostic, not gates.
- Artifact adequacy: passed. Each validation row stores stage-level scale-log,
  scale-logit, and hidden-preactivation summaries, with the run manifest bound
  to this plan.

Audit decision: `PASS_FOR_ONE_BOUNDED_Q20_TELEMETRY_DIAGNOSTIC`.

## Implementation Scope

1. Add compiled trainer telemetry for raw scale logits and hidden preactivations
   while preserving forward, logdet, gradient, and validation outputs.
2. Add runner summaries for per-stage scale logs, scale logits, and hidden
   preactivation tails/minima/maxima.
3. Add focused tests for telemetry shapes, finite values, stage ordering, and
   unchanged transport outputs.
4. Run CPU-hidden focused checks and a q=20 contract smoke.
5. Run one bounded GPU/XLA q=20 seed-a `(32,32)` diagnostic with the existing
   fixed-smoke parameters. Do not launch HMC.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
  --mode single-diagnostic --q 20 --batch-size 100 --hidden-layers 32,32 \
  --authorize-material-run --gpu-cap-seconds 7200 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-q20-scale-vs-elu-telemetry-2026-07-21/run-01
```

## Interpretation Rules

- High scale-logit tails with low hidden preactivation tails support a
  scale-head/optimization explanation, not an ELU explanation.
- High hidden preactivation tails in the stage that also saturates support a
  hidden-activation/conditioning contribution, but do not prove ELU is the
  root cause.
- High tails in both indicate interaction and require a controlled optimizer
  or scale-parameterization comparison.
- A normal telemetry run that still saturates remains a candidate training
  failure; no HMC or scientific claim follows.

## Planned Result Record

Create
`docs/plans/bayesfilter-ssl-lstm-neutra-scale-vs-elu-telemetry-result-2026-07-21.md`
with the command, artifact paths, stage tables, interpretation, decision table,
and post-run red-team note.
