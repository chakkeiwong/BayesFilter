# M13 particle-count ladder: N1008, N2016, N4032

## Research intent and evidence contract

Owner authorization (2026-10-10): test 2016 and 4032 to see whether more particles
help the completed N1008 range-bearing result. M13 remains the wrapped
constant-velocity model, T20, state4, observation2, parameter2. The target is
the model log marginal likelihood and its score. Each finite filter computes
its own fixed-design approximation and analytical total derivative.

Primary question: do the separately tuned N2016/N4032 versions of the new
covariance-guided defensive-mixture filter reduce conditional score error and
particle-design spread relative to the saved N1008 filter? The tuned previous
marginal-mixture LEDH is retained as a comparator at every N. The existing prior
proposal, global guide, and bootstrap/Fisher comparator remain the constructed
heuristic adversaries. Report truth and difficult parameter points separately
on each of data1704/18020/18021; no pooled ranking.

Use FP64/TF32-off GPU/XLA to match the baseline and isolate particle count.
Default FP32 is a separate unresolved scope, not silently changed or certified.
Eight paired particle designs per cell and arm, same seed schedule and datasets
as N1008. Same seed across N is an index coupling, not a claim that all draws
are nested: tensor shape changes alter the random layout. Frozen tuning and
unchanged references/data may be reused only for their own scope; each new N
gets fresh calibration and validation through the existing five-new/two-old
candidate grid and beta solver before held-out comparison.

Primary metrics: per-cell mean score-L2 error to the identical refined reference,
and RMS score spread (sqrt sum coordinate sample variances). Report paired
N-minus-N1008 mean error with nominal Student intervals and reference-sensitivity
expansion, label six-cell/two-count multiple and adaptive comparisons exploratory.
Variance and its ratios are descriptive with eight designs; do not assert a
statistically supported variance ranking from point estimates. Report score
centroid-reference discrepancy, log-likelihood mean error/SD, worst errors,
within-filter ESS, runtime/allocator memory and each raw replicate. The primary
criterion for scoped accuracy improvement is a negative reference-sensitive
paired interval with all required pairs valid; this is not default promotion.

Hard validity failures veto the affected candidate/comparison. Missing source,
model/data/score mismatch, faulty block parity/derivatives, or corrupted evidence
are continuation vetoes. Candidate rejection triggers the next already-planned
candidate or N, not research-direction rejection. No holdout retuning. A newly
observed shared implementation defect must be repaired and checked; affected
old evidence then cannot be used as an unchanged baseline.

Heuristic losses and unresolved reference bias veto broad promotion. Bootstrap
N16384/N65536 references with 16 designs remain finite-particle references;
no exact score, all-model result, posterior accuracy, convergence, or required
final N will be concluded. No HMC sampling is included. Results are preparatory
evidence within the surrogate-HMC objective.

## Required repair and skeptical audit

Pre-run audit found the new reset rejects N>3000. Simply lifting the CLI cap
would not execute the requested algorithm. Implement an exact tiled equivalent
of its finite Sinkhorn iterations, analytical directional recurrences, actual
column normalization, barycentres and Cholesky recolouring. Retain the dense
K=N path for N<=3000 without numerical changes. Use the shared non-overridable
chunk selector: N2016 K2016 and N4032 K2016. Do not substitute the historical
residual-injection Contract-E reset, change iteration semantics, add damping,
or relax moment/finite guards.

Verify call-chain wiring, dense/single-block parity at policy-compliant small N,
and actual multi-block N4032 parity using repeated four-atom distributions:
replication preserves source weights and uniform target masses, so grouped
finite Sinkhorn recurrences and reset tangents equal the small four-atom
reference. Also finite-difference a nondegenerate multivariate reset and rerun
the existing healthy LGSSM/KSC/shared reset checks. N1008 uses unchanged dense
arithmetic; confirm baseline numerical dependencies or bounded replay before
reusing its measurements. A small GPU preflight precedes serious runs.

The remaining audit passes once these checks pass: comparison targets and
precision match, references/data are shared, each N retunes, errors are judged
against the reference rather than proxy training loss, and heuristic/invalid
arms cannot erase valid primary pairs. The same old heldout datasets make this
a descriptive extension of a known diagnostic, not fresh confirmatory evidence.

## Defaults, assumptions and pre-mortem

- Horizon/points/data/design seeds: inherited exact comparison scope; preserve
  for controlled comparison, not evidence of generalization. Check data hashes.
- FP64: explicit reference exception matching baseline; no FP32 claim.
- Candidate grid/beta floor0.1/correction_steps0: frozen hypotheses from the
  prior reviewed diagnostic, justified there as defensive sampling with bounded
  calibration. Retune each N; no controls selected using heldout results.
  The particle ladder does not isolate pure N from the effect of retuning.
- Reset epsilon/finite iterations and previous-route damping/caps: unchanged
  tested hypotheses under the prior scope plan; variation retained in the
  existing tuning grid, no new numerics-altering protection is introduced.
- Eight designs: budgeted exploratory extension; noisy tail/variance estimates
  and limited dataset coverage are explicit. Extreme outliers remain reported.
- Variance ~1/N: hypothesis to assess, never assumed for the verdict. Persistent
  centroid/reference disagreement can reveal that particles alone do not help.
- Source/reference reuse: verify data and relevant algorithm hashes; the sole
  shared numerical edit is an N>3000 exact tiling extension with unchanged N1008
  path. Finite-program derivative checks are engineering evidence only.
- A run could pass while misleading us through reference bias, selection on
  already inspected datasets, or a higher N selecting a worse calibrated arm.
  Save frozen settings, report individual regimes, heuristics, failures and
  reference uncertainty; do not announce universal improvement.

## Execution, budget and results

One local GPU, serial workers, escalated access, verified memory growth,
TensorFlow/TFP/XLA; no package changes, external services or paid compute.
Bound this extension by 4 hours aggregate child wall time and 20 workers,
including focused checks and localized retries. A repaired failed launch uses
a fresh versioned directory; all failed evidence remains preserved. Stop on
budget exhaustion, persistent invalid harness, or absent valid scope tuning;
one failed particle count does not prevent the other valid planned count.

Master remains scripts/run_range_bearing_covariance_comparison.py; add only
bounded support for the ladder and read-only/post-run assembly. Commands use
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B followed by the absolute
script path and the existing run/worker/report actions. No shell environments,
pipelines or redirections; the master owns logs, environment and subprocesses.
Existing scoped approvals cover these forms; GPU jobs use escalation.

Versioned root: docs/plans/artifacts/ledh-range-bearing-particle-ladder-20261010-01/
with per-N child directories, plan/source hashes, exact commands, seeds, data
identity, worker timing/exit, validity, allocator memory, tuning and raw rows.
N1008 baseline: docs/plans/artifacts/ledh-range-bearing-covariance-20261009-07/.
Result: docs/plans/ledh-range-bearing-particle-ladder-results-2026-10-10.md.
Update docs/plans/ledh-main-merge-active-checkpoint.md after checks and runs.
The master should progress from repair checks through N2016 and N4032 and
aggregate the result without asking for routine phase approvals.


### Attempt 01 infrastructure repair

25 existing healthy/report regressions passed. The block preflight failed at
tracing because TensorArray.concat lost its known N dimension. Added explicit
static shapes, not looser invariants or changed numerical operations. Attempt
01 (32.64 worker seconds, two workers) is preserved; retry02 inherits its cost
and worker count under the same total four-hour/20-worker budget.
The selected device is the explicitly identified RTX4080SUPER GPU UUID, since
the current ordinal0 is a busy RTX5080. Memory growth remains enforced.
