# LEDH LGSSM XLA Value-Stage Instrumentation Plan

## Objective

Instrument the LGSSM LEDH value benchmark so an interrupted full-shape run can
distinguish XLA trace/compile plus first graph-call execution from host tensor
materialization.  The immediate question is whether the apparent `N=10000,T=50`
stall is Python-loop unrolling, XLA compilation, compiled graph execution, or
post-call materialization.

## Evidence Contract

| Field | Contract |
|---|---|
| Question | Where does the first full-size LGSSM LEDH value call spend time after the runner initializes GPU/XLA? |
| Baseline/comparator | Existing progress artifact that stayed at `initialized` for `N=10000,T=50`, plus a tiny trusted GPU smoke proving the new markers work. |
| Primary pass/fail criterion | The runner emits durable nonterminal progress artifacts for `value_call_started`, `value_call_returned`, `value_materialize_started`, and `value_materialize_completed` on a tiny GPU smoke. |
| Veto diagnostics | Marker names are absent or ambiguous; progress artifacts are terminal/admission artifacts; CPU-only evidence is mislabeled as GPU evidence; score/leaderboard readiness is claimed from marker diagnostics. |
| Explanatory diagnostics | Elapsed seconds in each progress artifact, GPU memory info, XLA log timestamps, and whether a larger diagnostic stops before or after `value_call_returned`. |
| Not concluded | No score correctness, same-scalar FD correctness, leaderboard admission, posterior correctness, HMC readiness, or runtime ranking is concluded. |
| Artifact | Progress JSON under `docs/plans/artifacts/` and this plan/result note. |

## Skeptical Audit Before Execution

- Wrong baseline risk: do not compare runtime to older non-instrumented artifacts
  as if they had the same stage markers.
- Proxy metric risk: marker completion only localizes the wait; it does not
  validate LEDH score or row admission.
- Hidden assumption risk: the active benchmark route must remain
  `streaming_batched_ledh_pfpf_ot_value_core_tf`, not the older Python-time-loop
  value core.
- Environment mismatch risk: GPU/XLA diagnostics require trusted/escalated
  execution; CPU tests are only regression checks.
- Artifact risk: an interrupted run must leave the latest stage in the progress
  JSON before the long call or materialization begins.

Audit status: pass for the instrumentation-first diagnostic.  The first GPU
run will be a tiny marker smoke only; no full `N=10000,T=50` run is authorized
by this plan until the marker smoke passes.

## Steps

1. Patch the benchmark to emit progress immediately before and after the first
   compiled value call and immediately before and after materialization.
2. Add a focused CPU-hidden unit test for the progress marker semantics.
3. Run the focused CPU regression test.
4. Run a trusted tiny GPU smoke with small `N`, `T`, and Sinkhorn settings.
5. Inspect the smoke artifact and decide whether a larger value-only diagnostic
   is now informative.

## Stop Conditions

- Stop if the CPU regression fails.
- Stop if the tiny GPU smoke does not produce a terminal artifact with all value
  stage markers.
- Stop if the smoke routes through a CPU device while claiming GPU evidence.
- Stop before interpreting a full-size timeout as scientific or score evidence.
