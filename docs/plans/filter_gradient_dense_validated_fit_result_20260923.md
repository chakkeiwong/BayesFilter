# Validated dense fitting result

The E5 validated-fitting boundary now encloses exact partition validation and
the existing fixed-center dense/factor fitting program in one reusable
TensorFlow/XLA call. It accepts explicit center, center score, padded offsets and
scores; active rows retain their original training/selection/audit extents.
Invalid finite inputs or copied rows return before fitting. The fitting output is
passed to the unchanged report formatter, which now lives in a shared helper;
its AST is identical to the former formatter except for the static audit-row
argument. No fitting or selection is repeated at the host boundary.

| Evidence | Result |
| --- | --- |
| CPU complete records |03408 D1 healthy and03409 D3 healthy pass all original fit, stability, selection, audit and lineage fields at unchanged comparisons.03410 audit veto,03411 incomplete stability caps, and03412 invalid-rank error all pass. |
| GPU complete records |03417 D1 healthy,03418 D3 healthy,03419 audit veto,03420 incomplete caps,03421 invalid rank and03422 enclosing/error order pass on the selected unshared GPU UUID with verified memory growth. |
| Enclosing loop |03414 CPU and03422 GPU reject a copied-row first iteration with error6, then run the original cloud with errors `[6, 0]`. Changed/return operands preserve one inner and outer trace and stable HLO. |
| Gradient boundary |03413 first exposed a zero-valued gradient edge through an added outer loop.03414/03422 place explicit `tf.stop_gradient` on completed outer results, restoring the original disconnected (`None`) public gradient behavior. The diagnostic reports zero maxima before the boundary. |
| No-use | Invalid center, score, and copied-row cases perform zero fitter calls. Valid cases perform one fitter call. A fit error cannot produce a usable covariance or scale. |
| Existing consumers |121 fixed-geometry checks pass on CPU03415 and had passed on GPU03404 against the same native fitter; policy03416 renews all129 checks. |
| Static guard |232 guarded sources,1333 exact exceptions, no new waiver. The four existing report-format exceptions moved to the extracted helper; no numerical-loop or NumPy exception was added. |

The preserved failures are part of the evidence.03413 was a harness assertion
about gradient disconnection, not a numerical mismatch; the retry identifies the
zero edge and installs the explicit public boundary. Earlier03400 remains the
subnormal identity defect fixed by the bitwise validator. No tolerance, solver,
selection, target callback, partition, RNG or scientific criterion changed.

The result is an execution qualification for the prepared dense-fitting
component. It does not qualify the external initializer's attempt loop, actual
seeded cloud generation, NPZ compatibility, public MacroFinance call chain,
actual target/transition callbacks, capacity under real data, terminal endpoint
inventory, or main merge. The E2 clipping-count proposal remains pending and
canonical LEDH rebuild remains excluded.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Promote validated fitter dependency to E5 composition | Full original records and no-use/error-order checks pass on CPU/GPU | No numerical or policy veto in tested scope | Actual external initializer still has Python attempts/cloud assembly and host serialization | Execute the bounded attempt-composition unit | Full initializer or DZ5 readiness |
| Preserve frozen derivative boundary | Original public geometry exposes no tape gradient |03413 diagnosed an outer-loop edge and03414/03422 repair passes | Other public callers may have separate derivative contracts | Renew affected consumers during composition | General gradient correctness |
| Keep campaign open | E1--E6 map still lists public integration and terminal gates | Main merge remains blocked | RNG, target transition and endpoint/cost evidence | Continue under remaining caps | Repository-wide policy closure or scientific admission |

The worker receipt is `artifacts/filter-gradient-repair-20260917/dense-validated-fit-checkpoint-03422.json`.
It binds manifests3408--3422, including the failed03413 attempt and retry,
with exact commands, sources and environments.
