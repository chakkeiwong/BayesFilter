# Conditional SIR slowdown: source and compiler trace

**Owner reset, 23 October 2026:** all cited N64 measurements and their performance
conclusions are disqualified. Source reasoning is not a replacement for fresh
N1008 measurements; see [replacement plan](ledh-sir-n1008-runtime-20261023.md).

23 October 2026. Worktree `ledh-iapf-execution-20260922`, branch
`ledh-refactor-with-policy-fix`, HEAD `90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`.

## Question and verification contract

Explain why the conditional arm became slower after the merge and why the latest
grouped-determinant result does not demonstrate recovery of pre-merge speed.
This is a source/artifact investigation, not another timing experiment or a
guide-quality comparison. No runtime implementation or numerical control changes.

Compare three explicit sources: the preserved pre-merge program in
`artifacts/ledh-sir-postmerge-replay-20261021/attempt-01/before-source`, the repaired
merged program in `artifacts/ledh-sir-merge-repair-20261022/candidate-02-final-source`,
and the grouped program in
`artifacts/ledh-sir-guard-placement-repair-20261022/candidate-01-source`.

The pass criterion is an executable source call-chain check, current-source
identity with the grouped snapshot, and reproducible inspection of the archived
optimized-HLO dependencies. A hash, wiring, scope, or compiler-dependency mismatch
blocks attribution until resolved. Existing paired timing intervals are supporting
evidence; source operation counts are explanatory, never measured runtime shares.
Stop on inconsistent artifacts. No new GPU calls, tuning, or timing retries.
Budget: one CPU-only static verification, under five minutes; the prior remaining
GPU-process allocation of 1141.424326 seconds is unchanged.

Skeptical audit: PASS for this bounded investigation. Comparing the latest source
only with the preceding repair cannot answer pre-merge recovery. Calling the
zero-direction endpoint value-only is also wrong: it calls the same analytical
score program. The archived workload is a diagnostic FP64 SIR scope with 64
particles, five observations and 24 flow substeps, not the default FP32/TF32
production scope. These settings are inherited comparison controls, not newly
tuned or promoted defaults. The identity and manifest checks expose stale sources
or changed controls before any interpretation. No heuristic-estimator comparison
is relevant to this engineering attribution; no estimator ranking is made.

## Source findings

The actual chain is `diagnose_sir_postmerge_runtime.build_endpoint` ->
`make_moment_endpoint` -> `build_backward_guided_ledh_kernel` ->
`_value_and_analytical_score_impl` -> `_flow_substeps_with_tangent`.
The current shared flow calls `_flow_substep_body_impl` for state advancement,
then `checked_log_abs_det_tangent` on the accumulated matrix/tangent group.
The endpoint consumes validity flags as well as moments, value and score.
It does not call the separate native-batch recurrence.

The timed labels are `plain` at physical-mixture weight zero, `conditional` at
weight one, and `physical` at weight 0.9. The timing factory supplies `guide=None`.
Thus the conditional timing is the current-observation Gaussian proposal; it is
not a benchmark of trained neural KDM or controlled-SMC guide evaluation.
This limits attribution of the timing results, not the coverage of the separate
full frozen replay.

For the conditional arm, `physical_mixture_draw` computes
`proposal_log = output_log` when beta is one. Its density does not use the LEDH
substep log determinants. At beta below one, the other component uses
`base_log -= log_det` and the corresponding derivative, so determinants are
needed numerically. The shared flow still executes before proposal selection.
Even beta one uses a runtime uniform comparison for selecting particle positions;
the source does not bypass the flow for that endpoint.

Before the merge, the compiled plain and conditional flow discarded unused
determinant calculations. The merge tied state validity to their factorization
and tangent validity to their derivative solve. That made these calculations
live even though the conditional proposal density did not need them. This is
additional numerical work, not just reporting a boolean. Archived optimized-HLO
dependency checks establish the dependency on individual state/tangent outputs.

The merge also replaced the innovation's direct Cholesky with `safe_cholesky`:
finite/symmetry checks, factor checks and safe selections remain within each
substep because the input depends on the moving particle. Their individual time
cost is not measured. The conditional proposal's observation conditioning,
covariance tangents and Gaussian-density algebra are ordinary additional work,
but `ledh_physical_proposal_tf.py` is byte-identical in the pre-merge, merged and
current sources. That unchanged code cannot by itself explain the regression.

The latest repair groups determinant matrices over 24 substeps before applying
the checked factorization. It still evaluates 5 x 24 x 64 = 7680 individual
9-by-9 matrix problems in this workload. For complete groups, the pivot loop is
now invoked five times on a batch of 24 x 64 matrices: 45 sequential batched
pivot iterations, not the preceding repair's 1080. State advancement still has
120 substeps and retains local innovation/state/tangent checks. Grouping changes
the schedule; it does not eliminate the determinant arithmetic or its acceptance
requirement. These are source-level counts, not timing measurements.

## Existing timing evidence

Ratios are numerator time divided by denominator time. The intervals are the
recorded approximate paired intervals, with the original shared-GPU limitations.

| Conditional endpoint | Repaired merged / pre-merge | Grouped / repaired merged |
| --- | ---: | ---: |
| Zero direction, same score program | 1.693 [1.503, 1.907] | 1.0032 [0.9753, 1.0309] |
| Nonzero direction | 1.382 [1.193, 1.602] | 0.9921 [0.9634, 1.0209] |

Sources: [pre-merge comparison](artifacts/ledh-sir-merge-repair-20261022/result.md),
and [grouped comparison](artifacts/ledh-sir-guard-placement-repair-20261022/result.md).
The second comparison finds no statistically supported conditional timing change
under its model; it does not establish exact equality or recovery relative to
pre-merge. Do not multiply ratios from different timing campaigns to manufacture
a current/pre-merge comparison.

The earlier [determinant-dependency intervention](artifacts/ledh-sir-merge-repair-20261022/attempt-09-determinant-cost/analysis.json)
reduced conditional times to 0.7168 [0.6048, 0.8495] and 0.6879 [0.5027, 0.9413]
of the guarded source. Optimized HLO dropped the determinant work, and all 216
healthy-input pairs matched exactly. This supports a material causal contribution
to runtime. It does not isolate exclusive kernel time. That ablation loses
singular-flow detection and remains rejected; it is not a proposed repair.

## Limits and next repair target

Established: changed determinant dependencies account for a material regression;
the latest timings use a different comparator; the timed conditional arm is not
a learned-guide evaluation. Not established: the exact runtime shares of each
guard/factorization, why grouping yields little conditional benefit, a direct
current/pre-merge timing ratio, or neural KDM evaluation cost. There are no usable
GPU-kernel profiler events in the archived investigation.

The next discriminating performance measurement should separate shared-flow,
validation and proposal-density costs in the actual grouped conditional endpoint,
retaining the same controls and validity behavior. Any proposal to omit a
determinant calculation needs an equivalent invertibility/derivative-validity
argument and fault checks. The prior design already records that a PSD proof for
predicted covariance is missing; `valid_predict` alone is not that proof.
Simply moving the final boolean to an outer scope does not remove the arithmetic
required to compute it. Static configuration checks can be hoisted, but matrices
depending on the evolving particles cannot be fully validated before the run.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept source-level explanation | Call chain, snapshot and archived HLO checks pass | Unsafe ablation remains rejected | Exclusive operation timings absent | Preserve guarded program; target actual conditional endpoint in future profiling | Full speed recovery or guide-quality improvement |
| Correct timing interpretation | Explicitly distinct comparators | No new default or estimator promotion | Current/pre-merge direct timing absent | Report both comparisons separately | Conditional endpoint equals pre-merge speed |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Unsafe guard removal is ineligible; earlier plain pre-merge numerical parity remains failed |
| Statistically supported ranking | Existing repaired/pre-merge conditional slowdown and guarded/ablated conditional differences only, under recorded paired assumptions |
| Descriptive-only differences | Static operation counts and possible remaining bottlenecks |
| Default-readiness | No change or new admission claim |
| Next evidence needed | Validity-preserving optimization and a direct matched comparison with operation-level attribution |

Post-investigation red team: shared-GPU contention limits timing attribution;
unchanged Gaussian conditioning is not evidence of a new bottleneck; calling all
grouping "batch mode" obscures the absence of an independent sample-batch axis.
The endpoint tensors have particle dimension 64, not a leading batch of datasets
or parameter points. A measured ranking of trained guides would require those
guides in the timed consumer. Numerical pre-merge parity is a separate unresolved
question and must not be declared repaired by the timing explanation.

Executable verification: PASS. Six relevant current source files match the
tested grouped snapshot; eight source call-chain edges and six archived optimized
HLO files pass the checks. The conditional-proposal module is unchanged across
the pre-merge, merged and current sources, and the timed guide is confirmed absent.
The CPU-only check took 0.843 seconds and used no GPU allocation.

Command: `CUDA_VISIBLE_DEVICES=-1 python3 -B docs/plans/artifacts/ledh-sir-conditional-slowdown-trace-20261023/verify_source.py`.
Evidence: [source-audit.json](artifacts/ledh-sir-conditional-slowdown-trace-20261023/source-audit.json).
