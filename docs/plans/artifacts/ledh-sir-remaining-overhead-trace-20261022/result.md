# Why the repaired SIR consumer remains slower

The main cause is a new dependency in the flow recurrence. The merged code
makes each updated particle depend on determinant validation and each updated
particle tangent depend on the determinant's derivative. Before the merge, XLA
discarded those calculations in the plain and conditional consumers. The repair
reduced their matrix dimension from 18 to 9 but did not remove that dependency.
The retained implementation therefore still does work the pre-merge consumer
did not execute.

This investigation changes no runtime code. It uses the actual frozen consumer
HLO, verifies 834 pre-merge and 980 retained package files against their source
indexes, and verifies all 980 retained files against the working tree. The timing
runner also matches its frozen auxiliary index. Baseline and candidate manifests
agree on inputs, controls, precision, XLA, TensorFlow, custom-op hash and GPU.
The scope is SIR, five observations, 64 particles, 18 states, 9 observations per
time point, FP64, TF32 off, 24 flow substeps, and unchanged OT (1,5). It is an
engineering diagnostic of this scope, not evidence for other scopes or guide
quality. The checked baseline postdates the LEDH historical-results invalidation.

## The dependency that changed

The executable source check follows
`scripts/diagnose_sir_postmerge_runtime.py:69` through
`make_moment_endpoint` and `build_backward_guided_ledh_kernel` to
`_value_and_analytical_score_impl`, `_flow_substeps_with_tangent` and
`_flow_substep_body_impl`. The endpoint uses the analytical score recurrence;
its zero-direction call is not a separate value-only implementation.

In `bayesfilter/highdim/ledh_canonical_score_tf.py:1463`, the checked determinant
returns its value, directional derivative and two validity flags. At lines 1470–1474,
the flags also depend on accumulated determinant and derivative finiteness.
At lines 1477–1482, these flags select the state and tangent passed to the next substep.
Consequently the checks are on the recurrence's critical path even when the
determinant is absent from the consumer's density formula. This is not merely
the cost of reporting an extra diagnostic at the end.

The HLO dependency checker selects individual tuple/fusion outputs, avoiding the
false inference that every output of a fused computation depends on every input.
It verifies that, in both plain and conditional consumers:

- The updated actual and auxiliary states depend on the determinant pivot loop.
- Both updated tangents also depend on its final triangular solve.
- The states themselves do not depend on that derivative solve.
- Neither determinant dependency exists in the corresponding pre-merge HLO.

The physical-mixture consumer already needed the determinant numerically before
the merge. Removing its validity dependency therefore leaves the factorization
live. This distinction explains why the same added guards have different
consequences across the three consumers.

## Why the current implementation is costly

`ledh_numerical_safety_tf.py:50` implements each checked determinant by partial
pivoting with a serial `tf.while_loop`. Each pivot permutes both the matrix and
derivative right-hand side, divides by the pivot, and updates both arrays. The
compiled body has four fusions and runs nine times. At line 79, a triangular
solve produces the determinant derivative, which is then checked for finiteness.

The measured consumer runs 120 batched flow substeps: five time points times 24
substeps. It consequently executes 1,080 batched pivot iterations, corresponding
to 7,680 individual 9×9 factorizations. The particles are batched; this is not a
Python loop over 7,680 particles. Reducing 18×18 to 9×9 helps but leaves the nested
loop, its dependencies and the derivative solve in place.

There is a second addition around the innovation Cholesky at
`ledh_canonical_score_tf.py:1373`. The old direct factorization became
`safe_cholesky`. Its full-triangle finite check, normalized symmetry check,
input selection, factor check and output selection are visible as seven extra
named fusion computations in each flow body. These checks remain in the
diagnostic determinant ablation. Their individual timing cost is not measured.

The following are compiled operations per flow-body definition, not runtime
percentages or measured GPU kernel durations. A nested loop repeats its own
body; the fusion column excludes that body.

| Consumer and source | Flow fusions | Nested determinant loop | GEMM calls | Cholesky calls | Triangular solves |
| --- | ---: | --- | ---: | ---: | ---: |
| Plain, pre-merge |27|None|14|1|6|
| Plain, retained repair |44|9 pivots, 4 fusions each|16|1|7|
| Plain, unsafe determinant ablation |36|None|14|1|6|
| Plain, checked Cholesky candidate |45|None|14|2|10|
| Conditional, pre-merge |23|None|12|1|6|
| Conditional, retained repair |40|9 pivots, 4 fusions each|14|1|7|

The checked Cholesky candidate removes the pivot loop but introduces another
factorization and more triangular solves. This explains the work exchanged by
that candidate; it does not establish its speed. Its prior parity check passed,
but three timing retention criteria remained inconclusive, so it was not kept.

The merge's small-matrix optimization is real but targets orders 1 and 2
(`ledh_small_matrix_tf.py:13`). The 9×9 innovation and 18×18 covariance operations
still use library factorizations/solves, as the HLO confirms. SIR's 2×2 moment
correction does use the specialization. It is incorrect to say either that the
optimization reaches SIR's main flow matrices or that it misses the whole
consumer. The separate native-batch recurrence is not called by this endpoint.

## What was ruled out, and what remains uncertain

The four ODE loop bodies and Sinkhorn loop have equal direct opcode counts
before and after the merge in all three consumers; their numerical controls are
also unchanged. This does not establish bitwise equality of their computations,
but rules out an explanation based on additional ODE/Sinkhorn iterations in
this replay. The scheduled-moment path is inactive. Both compared consumers are
compiled XLA routes with fixed signatures, not an eager-versus-compiled test.

Auxiliary states and their tangents already survive in the pre-merge flow carry.
They must not be presented as newly introduced calculations. The new carried
quantities in plain/conditional are the two determinant accumulators and two
validity masks. Other checks outside the flow remain present and are not assigned
an exclusive runtime share here.

Existing [attempt09](../ledh-sir-merge-repair-20261022/attempt-09-determinant-cost/analysis.json)
provides intervention evidence: removing determinant-validation dependencies
made the compiled determinant disappear in plain/conditional, while all 216
healthy-input output pairs remained bitwise equal. Three of six paired timing
intervals support reductions, with point estimates of 21–31%. That supports a
material causal cost of the new dependency. It does not measure an exclusive
kernel-time fraction or justify the ablation: singular-flow detection is lost,
so that implementation is rejected. Runtime guards remain intact.

Existing [attempt08](../ledh-sir-merge-repair-20261022/attempt-08-premerge-timing/analysis.json)
still supports slower plain nonzero and both conditional-direction calls versus
pre-merge, under its approximate paired interval model. The other three
conditions are inconclusive. Shared-GPU contention and absent usable profiler
events prevent complete cost attribution. These are scoped engineering timing
results; they establish no ranking of scientific estimators.

The score discrepancy is a separate unresolved numerical issue. Earlier traces
find a 4.16e-17 difference in first-observation weights and a 5.68e-14 state
difference after correction. By step two, covariance conditioning is about 1.2e9
and ESS about 2. Identical-input correction checks agree within tolerance, while
changed intermediate inputs reproduce the amplification. This supports a
sensitivity mechanism, not identification of the first differing floating-point
operation. Plain pre-merge numerical parity still fails; the direct four-input
score difference reaches 9.74046e-4. Do not call the programs numerically equivalent
or blame this discrepancy solely on the 2×2 solver. Full evidence and previously
repaired reset/correction defects are in the
[earlier result](../ledh-sir-merge-repair-20261022/result.md).

## Decisions and evidence status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the dependency explanation | Actual source chain and tuple-aware optimized-HLO checks pass | Frozen source/scope checks pass | Exclusive timing share unavailable | Optimize the checked flow factorization/solve schedule if further speed work is requested | Every remaining millisecond is explained |
| Retain candidate02 | Existing repair validation remains unchanged; 980 current files match | Unsafe ablation rejected; candidate03 timing retention not passed | Complete speed recovery unresolved | Preserve guards and source identity | A cheaper unchecked consumer is acceptable |
| Preserve the numerical-parity failure | Existing plain pre-merge comparison fails | No tolerance waiver | Earliest differing floating operation unresolved | A checked-reference sensitivity study before changing conditioning | Matching algebra proves matching finite-precision trajectories |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Source/HLO identity passes; unsafe guard removal remains rejected; plain pre-merge numerical parity remains failed |
| Statistically supported ranking | No new timing experiment; earlier scoped paired intervals support the stated slowdown and three ablation reductions only |
| Descriptive-only differences | Static fusion/library counts and unquantified validation costs |
| Default-readiness | No default, training, guard, tolerance or scientific promotion change |
| Next evidence needed | A guarded candidate with unchanged numerical checks and more precise timing; an independent reference for sensitivity attribution |

Red-team review: the strongest alternative explanation is that changed fusion
and scheduling, plus shared-device load, account for part of the observed
timing. The ablation cannot separate those effects from arithmetic work. The
weakest evidence is individual cost attribution outside the determinant chain.
A matched guarded implementation that removes this dependency cost without
reducing total runtime would overturn any claim that it explains all of the
remaining slowdown; no such comprehensive claim is made.

## Reproduction

Run in the LEDH worktree, using the existing frozen artifacts:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/ledh-sir-remaining-overhead-trace-20261022/trace_hlo.py > docs/plans/artifacts/ledh-sir-remaining-overhead-trace-20261022/hlo-trace.log 2>&1
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/ledh-sir-remaining-overhead-trace-20261022/check_dependencies.py > docs/plans/artifacts/ledh-sir-remaining-overhead-trace-20261022/dependency-checks.log 2>&1
```

Both diagnostic scripts use the standard library only. Their recorded analysis
wall times are 1.396s and 2.640s, within the 15-minute CPU process budget. No GPU
time was consumed. [hlo-trace.json](hlo-trace.json) and
[dependency-checks.json](dependency-checks.json) preserve commands, environment,
commit, source/HLO hashes, scope, input provenance, dependency checks and elapsed
time. Both complete successfully. Static dependency assertions verify this
trace; runtime tests were not repeated because the verified runtime is unchanged.

The existing GPU allocation remains 4,424.451901 seconds locally and 4,460.221424
seconds in the parent campaign. No task job is running. Branch:
`ledh-refactor-with-policy-fix`; HEAD `90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`.
No commit or push was performed; unrelated work is preserved.
