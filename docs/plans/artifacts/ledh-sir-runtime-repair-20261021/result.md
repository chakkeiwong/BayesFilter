# SIR runtime repair: completed and accepted within the tested scope

21 October 2026. Branch `ledh-refactor-with-policy-fix`, HEAD
`90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`; changes remain uncommitted.
The [reviewed plan](../../ledh-sir-runtime-repair-20261021.md) is complete.

The repaired shared flow takes approximately 10–18% less time than the frozen
merged QR implementation in the six endpoint/direction comparisons. All
2,496 replay output rows and 432 nonzero analytical directional checks pass
the unchanged numerical tolerances, with identical validity flags. This accepts
the implementation repair for the measured SIR scope. It does not establish
complete recovery of pre-merge speed, resolve the earlier plain-arm
pre/post-merge numerical discrepancy, or improve guide or score accuracy.

## Cause and implementation

The merge made determinant and tangent calculations necessary for validity
checks in plain and conditional consumers. Previously XLA eliminated their QR
work; physical-mixture consumers already needed it. The newly optimized batch
API is not called by these consumers and lacks their guide, physical-mixture,
moment-callback and analytical-recursion contract. Substituting it would change
the computation.

The repair retains these checks and replaces construction of Q with checked
partial-pivot elimination and a triangular solve on the same matrix. For
M = I + epsilon A, it computes log|det M| and
epsilon tr(M^{-1} dot A), exactly the mathematical quantities used previously.
The [terminal mathematical review](terminal-review.md) gives the derivation
and verifies the actual canonical-flow call chain. Native TensorFlow LU was
not supported by XLA_GPU_JIT in this environment, so one native TensorFlow
while-loop performs the elimination. There is no eager fallback, pfor,
regularization, new numerical threshold or autodiff score path.

Only two runtime package files differ from the frozen merged comparator:
`bayesfilter/highdim/ledh_numerical_safety_tf.py` and
`bayesfilter/highdim/ledh_canonical_score_tf.py`. The current package's
980 captured source files match the tested candidate snapshot. Failure handling
still separates invalid values from invalid forces; rejected inputs receive
placeholders only under false masks. Near-singular floating-point arithmetic
is not claimed to be an exact rank or condition-number oracle.

## Numerical and engineering evidence

The comparator is
`ledh-sir-postmerge-replay-20261021/attempt-01/merged-source`, with its completed
`after` replay as numerical reference. The frozen candidate is
[candidate-01-source/index.json](candidate-01-source/index.json).
Settings remain OT (1,5), terminal balance 32, N=64, T=5, 24 flow substeps,
FP64/TF32 off, exact chunk 64, and unchanged guide weights, fitted parameters,
data and seeds. Dataset seeds are 263043001 through 263043064, with four particle
replicates. Nine particle arms and three simple heuristics are retained.

The [complete comparison](comparison.json) records `repair_acceptance: true`:

| Check | Result |
| --- | --- |
| Complete frozen replay | 2,496 output rows and 432 nonzero analytical directional rows pass |
| Validity and finite outputs | No invalid rows, missing checks or changed flags |
| Fixed numerical tolerance | abs(error) <= 1e-8 + 1e-8 abs(reference), including raw moments |
| Largest fraction of allowed error | 0.002426, or 0.243% of tolerance |
| Maximum log-value difference | 6.14e-12 |
| Maximum nonzero analytical-score difference | 1.49e-10 |
| Maximum normalized-moment difference | 9.99e-12 |
| Source, inputs and controls | Hashes unchanged; all six replay endpoint trace counts equal one |
| Focused tests | 92 distinct passes, one inherited skip |

The tests comprise 16 GPU/XLA determinant, derivative, invalid-input and wiring
cases; 73 existing focused regression cases; and three CPU reporting-rejection
cases. The existing suite initially exposed a stale failure injector: it
overwrote a newly appended diagnostics dictionary instead of the validity item.
Correcting its index makes both failure-injection cases pass without changing
runtime source. The reporter now explicitly enforces this repair's raw-moment
tolerance rather than inheriting the older reporter's looser absolute tolerance.

Raw quantities have different scales. Their largest absolute block-loss
difference is 1.21e-5, but the largest strict error/tolerance ratio across raw
fields is 0.0001741. The largest raw-moment estimate difference is 1.64e-8.
Every comparison passes the fixed absolute-plus-relative rule; it would be
incorrect to describe all absolute differences as below 1e-8.

## Runtime evidence

Eight alternating blocks use four preselected datasets, one from each joint
observation/compartment regime, with endpoint and dataset order rotated.
Compilation is excluded. There are 384 timed calls and 216 paired output checks
including warm calls. All output comparisons pass; no individual dataset has
an observed mean runtime regression. Ratios below one mean less runtime.

| Consumer | Parameter direction | Repaired / merged QR | Approximate 95% interval |
| --- | --- | ---: | ---: |
| Plain | Zero | 0.837 | [0.770, 0.911] |
| Plain | Nonzero | 0.836 | [0.786, 0.888] |
| Conditional | Zero | 0.896 | [0.817, 0.983] |
| Conditional | Nonzero | 0.855 | [0.772, 0.946] |
| Physical mixture | Zero | 0.818 | [0.754, 0.887] |
| Physical mixture | Nonzero | 0.838 | [0.747, 0.940] |

All six intervals satisfy the predeclared criteria: an upper limit below one
for plain/conditional and at most 1.10 for physical mixtures, in both directions.
Intervals use Student-t with seven degrees of freedom on eight block means of
paired log ratios. Independent block errors are an approximation; this is
conditional evidence on four exposed inputs and one RTX 4080 SUPER, not a
universal runtime ranking. Per-regime differences remain descriptive.

Optimized consumer HLO contains live QR before the repair and none afterward
for all three endpoints. Both timing workers report allocator peak 10,033,152
bytes; the complete replay reports 70,575,872 bytes. Memory growth was verified
before initialization. Detailed timings, HLO and trace checks are in
[attempt-01-timing/analysis.json](attempt-01-timing/analysis.json).

## Decision and interpretation

| Decision item | Status |
| --- | --- |
| Decision | Accept the same-matrix runtime repair within the tested SIR scope |
| Primary criterion | Primitive checks, consumer parity and paired runtime criteria pass |
| Veto diagnostics | No repair veto; the existing scientific guide-promotion veto remains |
| Main uncertainty | Broader shapes, production FP32/TF32 consumers, near-singular arithmetic and timing dependence |
| Next justified action | Retain this implementation; use fresh scoped validation for broader runtime or scientific claims |
| Not concluded | Full pre-merge speed recovery, pre-merge plain parity, better moments/scores, HMC or default readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical, validity, provenance or resource veto in the repair; prior guide heuristic veto unchanged |
| Statistically supported ranking | Runtime intervals favor the repair in the six measured comparisons under the stated block assumptions |
| Descriptive-only differences | Per-regime timing means, allocation peaks and tiny numerical differences |
| Default-readiness | No new default or wider numerical-policy promotion; settings remain fixed |
| Next evidence needed | Fresh execution scopes for wider speed claims; untouched target-specific studies for guide or score-accuracy claims |

Prediction, observation-only and recursive-Gaussian heuristic comparisons are
retained. Both replay versions have `heuristic_dominance_verdict: promotion_veto`;
the veto lists match in all nine reported aggregate/conditional regimes. The
guide's previous negative moment conclusion is unchanged. This rejects a new
guide-promotion claim, not the validity of this runtime optimization or the LEDH
research direction.

The strongest alternative explanation for part of the runtime difference is
residual host-load/time dependence across only eight blocks. Alternation,
rotated order, paired output checks and consumer HLO support the measured
mechanism, but do not remove that limitation. A reproducible validity/parity
failure on an eligible consumer would overturn implementation acceptance for
that scope. The weakest generalization is from this small FP64 SIR workload to
other shapes, precisions and models; no such generalization is made.

The plan and terminal mathematical/code review were performed skeptically by
the executor. Independent Claude review was attempted but unavailable because
local authentication failed; independent approval is not claimed.

## Execution and remaining budget

[execution.md](execution.md) preserves exact commands, failed attempts and
repairs. [run-manifest.json](run-manifest.json) links the environment, source,
input and device provenance; [budget.json](budget.json) records all charges.
The full replay process used 1,241.93 seconds; its final analysis exited zero.
The repair used 1,821.33 of 7,200 allocated GPU-process seconds, including
probes, the failed primitive fixture run, tests, timing and replay. The parent
campaign has used 6,964.23 of 19,800 authorized seconds, leaving 12,835.77.
CPU reference/reporting checks are explicitly separate. All processes exited.
No retraining, retuning, tolerance change, commit or push was performed.
