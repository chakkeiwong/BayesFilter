# Five-iteration OT materiality diagnostic

The owner asks whether epsilon 1 with five Sinkhorn iterations materially changes
the likelihood and analytical score for the five active surrogate-HMC models.
This is a bounded diagnostic within the existing computation-repair master. It
does not change selected defaults or start posterior sampling.

## Evidence contract and research intent

Compare the fresh corrected-source selections in
`artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-advance-01`
against `reset_epsilon=1, reset_steps=5`. Keep flow steps, mixture weights,
correction controls, precision, model, observations, parameters, and particle
design identical within each pair. Use the actual batch-native analytical score
implementation, N4032/K2016, GPU/XLA, TF32 off. SIR retains the selected FP64
recursion/FP32 pair-kernel exception; the other models use FP32. Horizons are 50
for LGSSM/KSC/predator-prey/SIR and 20 for range bearing.

The primary screen is componentwise `abs(candidate-baseline)/max(1,abs(baseline))`
at most 0.01, separately for final log likelihood and every physical-parameter
score component. Report absolute changes, full score vectors, score norm changes,
and log-likelihood differences after subtracting the value at the nominal
parameter. The latter exposes changes hidden by a large additive likelihood
level. This 1% screen comes from the owner's materiality instruction; it is not
a posterior-equivalence theorem or a likelihood-MCSE estimate.

Non-finite output, incomplete filtering, or an existing kernel validity failure
is a hard veto for that arm, not a reason to skip other models. Record reset
covariance error and Sinkhorn row/column residuals as explanatory diagnostics.
Do not introduce a new Sinkhorn residual gate. Source drift, input mismatch,
GPU/memory-policy failure, or budget exhaustion veto continuation. A material
candidate difference is evidence about this candidate, not against OT/HMC.

Use the saved campaign dataset/design plus two fresh datasets (26101401 and
26101402), paired with designs seeded at data seed +700. Check these seeds are
disjoint from saved calibration/validation/claim seeds before running. Evaluate
six positions in the existing affine/multiplicative parameter chart: zero,
all -0.15, all -0.05, all +0.05, all +0.15, and alternating +0.15/-0.15. These
are local perturbations, not a posterior sample. Preserve the physical inputs
and their hashes. Nominal and symmetric perturbations guard against judging
only one favorable parameter point; two fresh datasets give limited replication.

No new tuning is performed. Frozen mixture weights isolate the requested change
but may not be optimal for it. The selected baseline is not a model oracle.
Because epsilon and iterations both change, this experiment cannot attribute an
effect to iteration count alone. Passing establishes local numerical similarity
only; model accuracy, statistical superiority, canonical admission, and HMC
convergence remain unproved. The heuristic-dominance gate is not invoked for a
method-quality ranking: this is an ablation against the identical implementation,
with no superiority/default claim. Model-reference comparisons remain separate.

## Budget, execution, and artifacts

Maximum three worker-wall GPU hours, charged to the existing 24-hour ledger,
with five serial workers and at most two localized retries. The campaign has
15.6344 hours and 21 attempts remaining before this diagnostic. Each worker has
at most 2,400 seconds, additionally capped by the three-hour total and the
remaining campaign budget. Use fresh output directories; retain failures.

Command (trusted GPU execution):

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_ot_five_iterations.py execute --output docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/ot-epsilon1-steps5-01
```

Use memory growth before device initialization, a stable B2 signature, the
existing analytical derivative recurrence, and no pfor or autodiff. Synchronize
calls before timing. Record compilation separately; take three warmed paired
timings with alternating arm order. Runtime differences are descriptive only.
Save per-group progress, full final values/scores, health traces, input/source
hashes and source snapshots, device/environment/command provenance, wall time,
per-model JSON, and terminal JSON/Markdown. Preserve all old selections.

## Pre-run skeptical audit and default audit

The baseline is the actual corrected-source selected configuration, loaded with
the repository scope/hash checks; it is not an old fixture default. The candidate
is the owner's explicit hypothesis, not a new default. Using the same precision
and frozen beta avoids confounding arithmetic or tuning with OT controls. The
existing local chart is a convenience coverage choice, explicitly insufficient
for posterior coverage. Fixed five iterations deliberately test a finite OT
program, not convergence of Sinkhorn. The existing recoloring can preserve moments
despite inaccurate transport; retain both moment and transport diagnostics.

Audit verdict: executable after focused checks of materiality aggregation,
invalid-arm classification, frozen-control replacement, and budget refusal.
The comparison answers the requested local sensitivity question. A misleading
pass remains possible outside the inspected parameters/datasets; a failure may
be a finite-program or numerical issue rather than evidence that OT is unsuitable.
Do not broaden this experiment into new tuning without reporting its results.

Execution review: eight focused CPU-only tests passed (materiality scaling,
single-component failure, invalid/non-finite arms, centered likelihood change,
control isolation, total budget, attempt exhaustion, worker overlap). The five
saved scopes have no source-hash mismatches. The 24-hour ledger has no active
workers. Trusted nvidia-smi finds the RTX 5080 and no compute-process entry.
The child will verify TensorFlow visibility and memory growth before execution.
The call chain uses the same make_batched_score_filter as the consumer, and
only the two requested OT controls differ. Audit passes for this diagnostic.

## Completed result

All five serial workers finished, with 90 valid paired rows and source identity
verified. The diagnostic charged 1,489.346674 seconds (24.8224 minutes); the
campaign has 15.2207 hours remaining. The actual TensorFlow device was the RTX
4080 SUPER with verified memory growth; the earlier nvidia-smi listing was not
used as the TensorFlow device identity.

LGSSM alone passes both 1% similarity screens. KSC, predator-prey, SIR and range
bearing exceed the score screen. All log-likelihood percentage changes are below
0.1%, but SIR's centered change reaches 1.40324 log units. This is an ablation
against selected finite programs, not evidence that the higher iteration count
is necessary or that either arm is more accurate against the model likelihood.
Only predator-prey holds epsilon unchanged. Existing selections remain unchanged.

Report and decision/inference tables:
artifacts/surrogate-hmc-cost-repair-24h-20261013-01/ot-epsilon1-steps5-01/report.md.
Machine-readable verification: the adjacent verification.json. All source hashes,
90 row identities, fixed non-OT controls, valid horizons and numerical checks
were independently reread after completion. No new posterior run was launched.
