# GenUT cap-report rounding mechanism

The FP32 cap-active disagreement is caused by the compiler's algebraic rewrite
of the cap formula followed by a discontinuous reporting predicate. It does
not require a different GenUT correction trajectory: the discrepancy reproduces
from identical saved pre-cap operands on CPU and GPU. The existing complete
cross-mode report gate remains failed; no tolerance or runtime source changed.

The mathematical cap is `x / (1 + (x/c)^p)^(1/p)`. Optimized XLA HLO computes
the capped value as `x * (1 + (x/c)^p)^(-1/p)`. Graph mode materializes the
rounded positive power before division. Both expressions represent the same
real-valued cap, but they round differently in FP32. Subtracting the result
from x and comparing it with 1e-7 turns a one-ULP difference into a one-count
difference. The runtime reports a finite-program displacement, not the exact
smooth-cap displacement.

| Evidence | Runs | Finding |
| --- | --- | --- |
| Original/candidate untouched vs instrumented CPU graph/XLA | 04111 | Every original return field remains exactly equal |
| Original GPU graph and candidate GPU graph/XLA instrumentation | 04110 | Every original return field remains exactly equal |
| Complete formula on identical saved operands | 04110/04111 | Graph 193 active coordinates, XLA 192, out of 216 |
| Division/subtraction from identical saved denominator | 04110/04111 | Restores graph active count in both modes |
| Optimization barrier after denominator, same saved CPU inputs | 04112 CPU / 04113 GPU | Barrier restores 193; ordinary XLA remains 192 |
| Optimized HLO | 04112/04113 | Ordinary XLA has negative-exponent power/multiply; barrier preserves power/divide |
| Independent 80-digit Decimal smooth cap | 04110/04111 | Disputed coordinate is below the 1e-7 threshold |

For GPU original-graph operand x=`0.2044350951910019`, coordinate [18,0]:

| Quantity | Value |
| --- | ---: |
| FP32 spacing at x | 1.4901161193847656e-8 |
| Exact smooth-cap displacement on that rounded input | 9.16430509331615e-8 |
| Rounded graph displacement | 1.0430812835693359e-7 |
| XLA formula displacement on the same input | 8.940696716308594e-8 |
| Executed FP32 threshold | 1.0000000116860974e-7 |

Thus graph's count is not an independent authority for the smooth-cap predicate.
XLA agrees with Decimal at this coordinate, but this one observation does not
establish globally accurate reporting or warrant declaring XLA the universal
reference. The exact finite graph and exact finite XLA programs disagree.
No candidate can match both frozen counts under the existing 2e-5 complete
record tolerance (their fractions differ by about 0.00463). A barrier in the
runtime would select graph rounding and fail the original XLA comparison; it
is retained only as a diagnostic mechanism probe.

The justified reporting repair direction is an explicitly defined diagnostic
target with a numerical-uncertainty/error outcome near the threshold, followed
by qualification of that target. Alternatively, retaining the exact
mode-dependent finite report requires recording cross-mode non-equivalence.
This result does not silently select either semantic change, replace the
threshold, or discard the failed gate. The user's instruction to report
ill-conditioned errors is consistent with testing an uncertainty outcome; it
does not justify inventing an uncalibrated uncertainty bound. A subsequent
reporting unit must derive/validate that bound before installing it.

04109 omitted `--device CPU` and therefore ran with the CLI's default GPU;
it is preserved and charged as a duplicate GPU diagnostic, never CPU evidence.
04110 is the correctly named GPU invocation, and 04111 explicitly hides GPUs.
The original FP32 GPU XLA compiler abort was not retried. No runtime barrier,
precision switch, TF32 change, numerical tolerance change or canonical LEDH
admission was introduced.

| Decision | Primary criterion | Veto status | Main uncertainty / next action | Not concluded |
| --- | --- | --- | --- | --- |
| Accept root-cause localization | Identical operands, exact instrumentation, Decimal and barrier agree | No instrumentation veto | Derive an uncertainty-aware report contract separately | Complete cross-mode equivalence |
| Continue reduction capacity tests | Rounding mechanism isolated | Report promotion veto remains | Measure untouched runtime at larger N,d | Full-target or canonical validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Strict report comparison still fails; no method-wide failure inferred |
| Statistically supported ranking | Not applicable to deterministic localization |
| Descriptive differences | One FP32 ULP crosses the report threshold |
| Default-readiness | No new admission |
| Next evidence | Validated reporting uncertainty and remaining target-capacity/public-consumer gates |

Skeptical result review: altered fusion due to added outputs was the strongest
alternative explanation; exact equality of every untouched return field and
the separate same-input primitive probe address it. Remaining uncertainty is
the report's intended numerical semantics and a reliable error bound over its
domain, not the observed mechanism. Raw operands, generated diagnostic sources
and HLO are preserved in runs 04109--04113.
