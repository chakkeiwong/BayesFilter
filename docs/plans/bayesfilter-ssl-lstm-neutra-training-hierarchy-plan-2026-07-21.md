# SSL-LSTM NeuTra Training-Hierarchy Repair Plan

Date: 2026-07-21  
Tier: 2 material numerical/GPU training-policy change  
Status: `AUDITED_FOR_EXECUTION`

## Research Intent Ledger

| Role | Contract |
| --- | --- |
| Main question | Can the q=20 NeuTra training loop continue through scale saturation while preserving hard numerical validity checks and applying a bounded repair? |
| Exact baseline | Existing q=20 `(32,32)`, batch-100, seed-a fixed-smoke run, validation every 250 steps, 2,000-step maximum, same target and support probes. |
| Candidate mechanism | Remove saturation from hard checkpoint eligibility and immediate-stop logic; evaluate paired validation loss at every finite/support-valid row, while treating `saturation_fraction > saturation_max` as a repair trigger that halves the learning rate once per plateau. |
| Primary criterion | The run must continue beyond the first saturation checkpoint without changing the transport math, and must terminate only by an ordinary plateau/max-step/resource/numerical rule. |
| Promotion criterion | None. This is one-seed policy validation; no HMC or posterior claim. |
| Hard vetoes | Nonfinite values, failed round-trip, failed support radius, invalid serialization/reload, GPU/XLA/memory-growth failure, corrupted artifacts, or resource cap. |
| Repair trigger | Saturation above the diagnostic threshold; trigger one learning-rate reduction when no repair has yet occurred in the current plateau. Loss improvement remains eligible to update the best state even when saturation is present. |
| Continuation veto | Any evidence that the policy change alters forward/logdet/gradient semantics, allows a numerically invalid checkpoint to become best, or prevents deterministic resume. |
| Explanatory diagnostics | Aggregate and stage saturation, raw scale-logit tails, hidden preactivation tails, loss, learning-rate path, clipping, runtime, and terminal reason. |
| Nonclaims | No claim that saturation is harmless in production, no architecture ranking, no HMC readiness, no posterior correctness, no optimal learning rate, and no default scientific validity. |

## Interruption-Recovery Amendment

The loss-first follow-up was externally interrupted after writing a valid
step-1250 joint checkpoint. The controller did not stop: its state remained
`running`, with `stop_reason=null`, `best_step=1250`, and `max_steps=2000`.
The harness nevertheless could not resume because it required a graceful
resource-stop receipt and outer summary.

The amended mechanism is:

1. Catch `SIGTERM` and `SIGINT` as a deferred interruption request. At the
   next safe boundary, write an interruption receipt bound to the latest
   complete joint checkpoint and exit with status `INTERRUPTED`.
2. If a process disappears before it can write that receipt, permit explicit
   `--resume` recovery from the latest checkpoint referenced by `progress.json`
   only after verifying its file hash, joint checkpoint hash, stream, trainer
   step, controller observation step, best-state hash, and restored
   trainer/controller configuration.
3. Resume from `last_program_step + 1`; never infer convergence or a scientific
   result from an orphaned `RUNNING` progress file.
4. Launch the resumed GPU process as a durable user service so its lifetime is
   not coupled to one interactive tool call.

The existing step-1250 checkpoint is a legacy orphan-recovery case: its state
and artifact hashes are available, but the older progress schema did not store
source hashes. Recovery is allowed here because the user explicitly requested
continuation, trainer/controller configuration restoration is exact, and the
only intervening runner edits add recovery mechanics rather than changing the
target, transport, optimizer, seeds, or numerical path. This exception cannot
support a source-identity or broad reproducibility claim.

## Policy Hierarchy

1. **Hard validity vetoes:** nonfinite computation, failed inverse round trip,
   failed moderate-shell support, serialization/reload failure, GPU/XLA or
   memory-growth failure, and resource limits.
2. **Repair trigger:** saturation above `saturation_max`. It is recorded and
   causes a one-time learning-rate reduction for the current plateau; it does
   not by itself make the transport invalid, reject a scientific direction, or
   terminate training.
3. **Checkpoint selection:** a checkpoint must pass hard validity checks. A
   saturated row may become the best state when its paired validation loss
   improves; saturation is recorded alongside that state and does not veto
   export. This is a training-selection rule, not a claim that saturation is
   harmless.
4. **Ordinary training stops:** plateau after the configured repair window and
   maximum steps. These describe optimization termination and are distinct from
   validity vetoes.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
| --- | --- | --- | --- |
| Existing `saturation_max=0.05` | Inherited diagnostic threshold; not mathematically derived | Preserve comparability while changing only its role | Threshold may be poorly calibrated; retain stage telemetry and do not call it a validity threshold |
| Halve learning rate once per plateau | Existing reviewed repair policy | Saturation indicates weak scale-head gradients; a bounded LR reduction is reversible | Could hide a target/geometry issue; compare loss, support, and later saturation |
| Preserve the loss-improving best state during LR repair | Objective-first training policy | Do not discard an observed paired-loss improvement merely because a diagnostic threshold fired | A lower loss can still be overfit or poorly supported; support and downstream checks remain required |
| Preserve max/plateau stops | Existing controller contract | Avoid converting a diagnostic into an unbounded run | A successful continuation is not evidence of convergence |
| One seed and fixed-smoke parameters | Matched prior diagnostic | Smallest discriminating test of policy semantics | Descriptive only; no robustness or ranking claim |

## Skeptical Pre-Execution Audit

- Wrong baseline: passed; target, seeds, batch, architecture, optimizer values,
  validation cadence, and math remain unchanged.
- Proxy promotion: passed; saturation remains diagnostic/repair evidence and is
  not a correctness or HMC admission criterion.
- Missing stop: passed; max-step, plateau, resource, nonfinite, support, and
  round-trip stops remain explicit.
- Unfair comparison: passed; the first test is the same q=20 seed-a baseline,
  differing only in controller policy.
- Hidden assumptions: recorded above; `0.05` is retained only for a repair
  trigger and is not presented as a statistical boundary.
- Artifact adequacy: passed; action payloads record repair kind, saturation,
  learning-rate reductions, paired loss comparisons, controller state, and
  full validation history.

Audit decision: `PASS_FOR_IMPLEMENTATION_AND_ONE_BOUNDED_Q20_POLICY_TEST`.

## Implementation Scope

1. Update `NeuTraPlateauController` so saturation is no longer a hard
   eligibility veto or immediate stop, and so paired loss is evaluated before
   the saturation repair action is emitted.
2. Add an explicit saturation-repair action and preserve deterministic state
   hashing/resume semantics.
3. Update complexity-runner result classification and manifests so saturation
   is recorded as a repair event, not a candidate-invalidity veto.
4. Update focused controller tests and add regression coverage that saturated
   observations continue, trigger one repair, and can replace the prior state
   when paired loss improves.
5. Run CPU-hidden focused tests and a q=20 contract smoke.
6. Run one bounded trusted GPU/XLA q=20 seed-a diagnostic with the unchanged
   fixed-smoke parameters. Do not launch HMC.
7. Add signal-safe deferred interruption receipts and verified orphan recovery;
   test both paths, then resume the verified step-1250 checkpoint with an
   additional 3,600-second cap. Based on checkpoint timestamps, the original
   partial execution used about 1,850 seconds, so the cumulative observed-plus-
   authorized ceiling remains below the original 7,200-second plan cap.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Question | Does the new hierarchy continue after saturation and apply the intended repair without weakening numerical validity? |
| Comparator | Prior artifact `ssl-lstm-q20-scale-vs-elu-telemetry-2026-07-21/run-01`, which stopped at step 750 on saturation. |
| Pass criterion | New run reaches a later validation step than 750 or an independent hard numerical/plateau/max/resource stop, records a saturation repair action, and passes support/round-trip/reload checks. |
| Vetoes | Any invalid artifact, nonfinite value, support/round-trip failure, unexpected math drift, GPU policy failure, or resource stop before the first post-saturation validation. |
| Explanatory only | Loss, saturation path, learning-rate path, runtime, and whether the final candidate would be scientifically useful. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-training-hierarchy-2026-07-21/run-01/` plus a result note at the matching `docs/plans` path. |
| Nonclaim | A continued run does not establish convergence, posterior correctness, HMC readiness, or that the saturation threshold is optimal. |

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py \
  --mode single-diagnostic --q 20 --batch-size 100 --hidden-layers 32,32 \
  --authorize-material-run --gpu-cap-seconds 7200 \
  --params-json docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/fixed-smoke-params.json \
  --output-root docs/plans/artifacts/ssl-lstm-q20-training-hierarchy-2026-07-21/run-01
```

## Stop And Interpretation Rules

- Stop immediately for hard validity, GPU policy, or resource failures.
- Stop at max steps or the ordinary post-repair plateau window and classify
  those as optimization stops.
- If saturation occurs and the run continues, interpret that as evidence that
  the hierarchy is operational, not evidence that the trained transport is
  correct or converged.
- If the run still becomes numerically invalid after repair, classify the
  repair as unsuccessful without reverting the policy change.

## Planned Result Record

Create
`docs/plans/bayesfilter-ssl-lstm-neutra-training-hierarchy-result-2026-07-21.md`
with exact action transitions, saturation and learning-rate paths, validity
checks, decision/inference tables, and a post-run red-team note.
