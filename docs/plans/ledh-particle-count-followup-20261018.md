# Final paired particle-count diagnostic, 18 October 2026

## Question and intent

Does increasing the positive-tail particle cloud from 64 to 128 reduce the
remaining score error under the same frozen messages and consumer? The preceding
32/64 diagnostic is complete and audited: 768 rows, 4,608 trace checks and 16
finite-difference checks. All 14 simultaneous contrasts remain unresolved.
Exact messages still have residual error, so more fitting capacity is not the
next discriminating repair. Its results are in
`artifacts/ledh-particle-count-stress-20261018/attempt-01/analysis.json`.

This is a conditional mechanism diagnostic, not a tuning or admission run.
Keep eight saved fits, both observations `(0.2, 5)`, parameters `(1.5, 0)`,
the analytical recursive frozen-guide score, and all numerical controls fixed.
Use twelve fresh nested clouds per fit with seeds `862800 + 100*c + r`.
The smaller cloud is the prefix of the larger cloud; chunks equal particle
counts, as required. The eight arms give 1,536 rows and 16 derivative checks.

## Evidence contract and diagnostic roles

- Primary diagnostic: paired changes in squared error of score coordinate 1,
  against an independently checked numerical reference. Each product consumer
  at 128 particles is compared with itself at 64 and with all six simple arms
  at 128: 14 predeclared contrasts. Average the twelve cloud differences within
  each fit, then use paired t intervals with seven degrees of freedom and
  Bonferroni 95% simultaneous coverage. Independence across saved-fit groups
  and approximate t calibration are assumptions, not a population guarantee.
- Constructed heuristic adversaries: unguided identity (no flow), unguided
  global (shared affine flow), unguided local (particle-local flow), unguided
  equal mixture (no mixture selection), Gaussian-future global (simple
  lookahead), and Gaussian-future equal mixture (lookahead without selection).
  Evaluate every arm separately at both particle counts and in both score
  coordinates, without pooling the positive-tail situation with other data.
- Promotion veto: descriptive squared-error loss against any heuristic, or
  uncertainty that does not establish the proposed improvement. These cannot
  become continuation vetoes. Passing this diagnostic cannot promote a method.
- Continuation veto: nonfinite/invalid traces, wrong data/score route, failed
  reference or derivative checks, incomplete coverage, source/input drift, or
  exhausted campaign resources. Guard failures are not scientific rejections.
- Explanatory only: bias, variance, runtime, allocator statistics, individual
  seeds and coordinate 0. Do not select settings on these observations.
- Repair trigger: exact and fitted messages both retaining error motivates
  isolating the current-observation approximation and reset in a separately
  budgeted next stage; a mere heuristic loss does not reject guidance itself.
- Do not conclude count convergence, general improvement, fresh-fit or
  population ranking, canonical LEDH conformance, default readiness or HMC
  readiness. This is an FP64 GPU/XLA diagnostic exception, not the TF32 default.

## Defaults, assumptions and pre-mortem

| Choice | Provenance and status | Reason, risk and earliest diagnostic |
| --- | --- | --- |
| Eight saved fits | Previous audited mechanism study; fixed conditioning inputs | Isolates the consumer; cannot estimate new-fit generalization. Verify copied fit hashes and seeds. |
| Positive-tail data and two observations | Previous unresolved stress situation | Targets the observed failure; cannot support other regimes. Check reference/data identity before evaluation. |
| 64/128 particles, twelve clouds | New bounded diagnostic design | Tests particle sensitivity with more replications; may remain underpowered. Report all simultaneous intervals. |
| OT `(epsilon, steps)=(1,5)` and correction controls | Frozen diagnostic comparator, not a tuned default | Holds the finite program fixed. Caps and ridges are not newly calibrated or promoted; retain validity/margin diagnostics and derivative checks. No setting is silently altered. |
| Product moments and exact seven-component message | Existing optional consumer and independent control | Separates fitting error from consumer error, not reset from flow. Inspect both products rather than assuming exact guidance eliminates all error. |
| FP64 GPU, XLA on, TF32 off | Explicit reference/diagnostic exception | Reduces numerical confounding; says nothing about TF32 production. Verify memory growth, placement, finite traces and source closure. |
| Cluster-t intervals | Same predeclared conditional analysis | Few clusters and skewed errors can miscalibrate intervals; retain cluster data and report unresolved contrasts without rankings. |

The run could succeed while misleading us if a saved-fit conditional contrast
were described as a fresh-fit confirmation, if a descriptive reduction became
a ranking, or if unchanged but uncalibrated safety controls were called a new
default. The contract excludes all three. It could fail for harness or budget
reasons rather than for scientific reasons; coverage, source hashes, references
and trace checks distinguish those cases.

## Budget, execution and automatic refresh

Parent: `ledh-message-consumer-continuation-20261018.md`. Its first two attempts
used 646.162868663 of 1,200 GPU-process seconds and two of three launches.
CPU reference/analysis used 3.464404211 of 180 seconds. This is the third and
final attempt, capped at 500 seconds within the 553.837131337 seconds remaining.
No unused budget from older campaigns transfers. Preserve unique outputs.
Localized harness fixes may use the remaining parent resources, but a fourth
launch or expanded scientific scope requires a new campaign decision.

Working tree: `ledh-iapf-execution-20260922`, branch
`ledh-refactor-with-policy-fix`; Python environment `tftwogpu`. Before editing,
preserve the preceding runner/analyzer in the artifact root's
`source-before-refresh/`. Only a fixed reviewed follow-up design is added; the
scientific kernels are shared unchanged.

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_particle_count_stress_diagnostic.py --followup --budget-seconds 500 --output docs/plans/artifacts/ledh-particle-count-followup-20261018/attempt-01
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/analyze_ledh_particle_count_stress_diagnostic.py --run docs/plans/artifacts/ledh-particle-count-followup-20261018/attempt-01
```

Run the first command with trusted GPU permissions; the second deliberately
hides GPUs. Preserve logs beside the attempt directory. The manifest records
commit, dirty-source hashes, command, data, seeds, device/memory policy,
environment, wall time, plan and result. Save a result note with decision and
inference-status tables. Then refresh the checkpoint, both master programs
and the manuscript. Audit/repair/refresh is automatic within the agreed scope;
phase boundaries alone never create a permission request.

## Skeptical pre-execution audit

PASS for this bounded diagnostic: same-data same-consumer comparisons and six
constructed heuristics avoid a weak baseline; score error, not guide fit loss,
is the measured target; count/chunk pairing is explicit; GPU/XLA and analytical
score identities are checked; reference and validity failures veto evidence;
seed reuse and completed-source snapshots are accounted for; the final-attempt
cap prevents an open-ended ladder. Remaining weaknesses are conditional saved
fits, few independent clusters, uncalibrated frozen controls and no isolation
of reset from flow. They prohibit promotion, not execution of this diagnostic.
