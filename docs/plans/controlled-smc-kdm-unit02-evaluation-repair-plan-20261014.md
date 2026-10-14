# Unit02 initialization: evaluation-only repair

14 October 2026, Hong Kong. Continuation under the existing unit02 initialization
plan and budget; no new training, scientific target, or acceptance criterion.

## Failure and retained evidence

Attempt01 completed 270 MDPF epochs with source early stopping (best epoch174),
21330 updates, 3782 training checks and 70512 terminal source-controller checks,
all passing. The old-start replay passed1863 checks with zero loss difference.
Eight-draw author validation passed: author6.21585142, oldTF6.72928978,
author-startTF6.31432726. These fixed-checkpoint results are descriptive.

Endpoint execution failed before TensorFlow import: its reused input checker
required four unit02 checkpoints, although the reviewed intervention calls for
three MDPF checkpoints. The exporter already produced the correct three models.
This is a harness-scope mismatch, not a numerical or author-algorithm failure.

## Evidence contract and intent

Question: complete the already declared native T150 evaluation to test whether
the initial measurement-model substitution reduces unit02's saved NLL gap.
Comparator, data, model weights, N25, T150,256 rows,batch64 and two filter seeds
202610082/202610083 remain those in the original plan.

Promotion criterion for this diagnostic is valid completion of the three-model
evaluation, independent author metric replay and unchanged baseline replay.
Gap reduction in both validation and endpoint NLL supports the within-unit
initialization explanation. Finite unfavorable outcomes remain reportable.
This cannot establish whole-KDM completion or population noninferiority.

Continuation vetoes: changed trained checkpoint or runtime, changed scientific
inputs, missing source audit, nonfinite output, incomplete row/model coverage,
trace count other than one per exercised signature, failed author metric replay,
baseline NLL/RMSE disagreement beyond the original1e-5 absolute tolerance, or
resource exhaustion. Such failures invalidate interpretation and trigger a
localized repair within the unchanged budget. Loss/RMSE contrasts and runtime
are explanatory; no criterion is tuned to these results.

The unchanged saved TF model is the cheap control for this causal question.
Adding other filtering algorithms would not test this harness repair.

## Repair and focused checks

1. Give the exporter an explicit evaluation_scope and require the evaluator to
   check the exact three MDPF identities for unit02_mdpf_initialization. Preserve
   the legacy recent_checkpoints scope with its four/five exact identities.
2. Check both saved legacy inputs and the planned three-model input. Reject a
   missing scope, duplicate/wrong model, wrong unit or unknown scope. These are
   small standard-library checks, before TensorFlow import.
3. Reuse frozen training, author validation and terminal audit. Check their
   hashes, training provenance and all unchanged numerical runtime files.
4. Fresh attempt02 regenerates inputs. They must equal attempt01 inputs exactly
   after removing the new scope field. Run GPU endpoint then CPU author metrics.
5. Preserve manifest, logs, source snapshots, machine decision and result note;
   refresh master/checkpoint. Never relaunch completed training.

## Assumptions and review

The count depended on an earlier evaluation scope, not algorithm mathematics.
A generic len(arms)>=3 would hide missing models; explicit identities prevent
that. Source/controller settings and numerical protection remain frozen; their
original derivation, provenance and risk checks remain in the parent plan.
The new scope is metadata only, verified by exact equality of every other input.
A stale checkpoint could misattribute improvement; hashes and saved audits veto it.
Two endpoint seeds and eight validation draws remain conditional diagnostics,
not independent training replications. A small residual gap does not prove exact
equality or furnish a population bound.

Skeptical audit passes: the exact old/source baselines are retained; no proxy is
promoted, no tuning or seed selection occurs, and success requires the actual
downstream endpoint. The regression checks distinguish this harness fix from a
changed evaluation. Strongest alternative explanation for a favorable result
remains random training trajectory sensitivity, as recorded in the parent plan.

## Execution and budget

Escalated exact command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_unit02_evaluation_repair.py

Read-only --preflight first. GPU uses tftwogpu, the pinned RTX5080, XLA,FP32,
TF32off and verified memory growth. CPU reference uses tf-gpu2 with GPU hidden.
New output: unit02-initialization-20261013/attempt02 in the parent artifact root.
At most this second attempt; cap900GPU/300CPU seconds within the parent's
21600GPU/3600CPU seconds and the existing campaign balance. Attempt01 used
5115.153827GPU/90.740936CPU seconds. The supervisor records progress every30s,
kills only its own child on timeout, and charges this retry exactly once.
No new allocation, new approval ceremony or environment mutation is required.

Result: controlled-smc-kdm-unit02-evaluation-repair-result-20261014.md.
Pre-mortem: successful execution alone could hide a missing comparator; exact
identity checks and six output rows prevent that. Preserve failure artifacts.

## Implementation review before execution

All nine scope regression cases pass using the evaluator's actual input-check
helper, including the two saved legacy inputs, the declared intervention, and
six invalid-scope/model-set cases. Both changed modules and the recovery
supervisor parse. Read-only preflight passes: the complete 270-epoch checkpoint
and 70512-check source audit are intact, unchanged runtime files match the
original source snapshots, and 71819.978301 GPU / 205468.572852 CPU seconds
remain in the existing allocation. The retry uses at most 900 GPU / 300 CPU
seconds. The numerical kernels, model parameters, data and seeds are unchanged.
The recovery script contains no training launch and compares regenerated inputs
field-for-field against the prior export. Reviewed for execution.
