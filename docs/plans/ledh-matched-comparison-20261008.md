# Matched T=50 comparison, 2026-10-08

The question is whether the covariance-guided mixture candidate loses likelihood
or score accuracy relative to the previously tested filter when the experimental
conditions agree. The owner requested this rerun after the N=128 comparison used
different observations and a different LGSSM parameterization.

## Evidence contract and intent

Use T=50, N=1008, FP64 GPU/XLA, TF32 disabled, memory growth, the P44 d=3 LGSSM,
the complete seven-component KSC model, predator–prey, and SIR d=18. Use data seed
26100611. Read the saved nonlinear observations verbatim; regenerate LGSSM/KSC
with their original simulator and verify the protected seed-261006201 replay.
Main designs are 261006101–104 for both methods; LGSSM/KSC/PP also replay seed
261006201 against the saved October 6 results. Both methods receive identical
initial normals and process normals from the old `random_inputs` authority.
The candidate additionally requires categorical branch/ancestor uniforms; these
are independent fixed draws, explicitly recorded, with no counterpart in the
old IID method. Equal seeds alone are not evidence of equal random inputs.

The comparator is the old `iid_dual_cap`, guarded-pairwise, normal-quantile-reset
route, ancestor weights for LGSSM/KSC and marginal weights for nonlinear models.
Its controls are frozen at the October 6 settings. The candidate retains the
previous implementation controls (16 flow steps, 40 reset steps, epsilon=1,
no optional moment correction). Its beta is calibrated afresh at this N and
target using independent data seeds 26100820/21 and the existing pilot solver.
Calibration does not inspect comparison errors. These algorithm-specific
controls differ by design; this compares complete implementations, not an
isolated importance-weight change or optimally tuned methods.

Primary evidence is per-design absolute likelihood error and score-coordinate
errors against the SAME reference, including score L2 error and paired
new-minus-old error differences. Report all raw values and scores. Exact Kalman
and checked refined KSC references are recalculated. The saved N=131072,
four-replication bootstrap/Fisher references and Zhao–Cui quadratic references
are reused only with identical saved nonlinear data and target identities.
Bootstrap uncertainty and finite-particle bias remain limitations; Zhao–Cui is
an independently labelled numerical comparator, not an exact oracle.

Four-design means and standard errors are descriptive. Report paired t intervals
(df=3), explicitly conditional on one dataset and on the numerical reference.
No across-model or default promotion follows. A loss against a simple comparator
is a promotion veto, not a continuation veto. The constructed heuristic set is
the old filter (existing practitioner method), same-N bootstrap (unguided
importance sampling), and the exact/refined Gaussian reference where available;
the nonlinear high-N bootstrap additionally measures reference sensitivity.
Also run the old covariance-only reset (normal-quantile design retained, marginal and pairwise correction iterations set to zero) as a simpler comparator for every model. Evaluate separately for each model, not by pooling model errors. No candidate
is selected against this falsification set.

## Audits, stop conditions, and budget

Default audit: N, T, data, parameterization, old controls and normal inputs come
from the saved campaign. Their adequacy is a baseline hypothesis, not a new
default. Candidate flow/reset settings are frozen hypotheses from its last run;
failure may reflect those choices. Beta's pilot objective targets weight
variability, not score accuracy. FP64 is an explicit reference exception to the
production FP32/TF32 direction. GPU 1 is used to avoid the occupied GPU 0;
both new arms run on this same device. Replay tolerances allow hardware rounding:
1e-8*(1+abs(value)) and 1e-6*(1+max(abs(score))) respectively.

Pre-mortem: a successful command could compare different targets, random tensors,
initial timing, or bootstrap data; input hashes, target IDs and protected replay
must exclude these errors before interpretation. A score can correctly
differentiate a poor finite filter, so derivative consistency is engineering
evidence only. Singular resets or invalid beta contexts are candidate failures;
retain them and complete other models. A failed old replay, mismatched target or
corrupt/missing comparison input stops that scope for localized repair.

Skeptical review: the earlier comparison's N/target/data mismatch is removed.
No inherited beta is promoted to the new scope, no reference is used for tuning,
and no speed or ESS ranking substitutes for reference error. Only one dataset
per model and four designs remain weak statistical evidence. The plan therefore
supports a matched diagnostic comparison, not superiority or broad accuracy.

Budget: at most 4 hours of aggregate worker time and 24 launched processes,
including calibration, old/new evaluation, focused checks and localized retries.
Use existing shared numerical implementations, no model or filter fork. Maximum
individual worker is 1800 seconds; preserve completed rows incrementally. Stop
at the budget, recording incomplete cells without treating them as successes.
No package installation, paid compute, external publication, or default change.

Commands: the matched driver `docs/benchmarks/run_ledh_matched_comparison.py`
provides `run`, `worker`, and `report`; worker subprocesses run under the approved
Python environment with CUDA_VISIBLE_DEVICES=1, TF_FORCE_GPU_ALLOW_GROWTH=true,
and bounded TensorFlow CPU threads. Exact argv, source hashes, environment,
seeds, timings and GPU memory policy are preserved in its manifest/attempts.
Output root: `docs/plans/artifacts/ledh-matched-comparison-20261008-01/`.
Result note: `docs/benchmarks/ledh-matched-comparison-results-20261008.md`.
