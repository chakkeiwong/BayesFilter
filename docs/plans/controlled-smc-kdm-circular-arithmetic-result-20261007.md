# Circular arithmetic localization result — 7 October 2026

The discrepancy is not removed by matching the remaining Normal arithmetic and
dimension-addition choices. Every one of eight CPU float32 arms still fails
the preserved comparison bound. All 928 float64 endpoint checks pass, and all
36 independently assembled weight/location/bandwidth derivative checks pass
against float64 autodiff on the exact cast float32 fixtures. Both replications
remain incomplete.

The measured target is the author's stopped-sample IWSG density derivative and
its normalized-weight/posterior-loss consumers. The algebraic gradients agree
on these bounded fixtures; the finite float32 results differ. This is evidence
for cancellation and finite operation ordering, not proof that every numerical
difference is harmless or that the full training translation is qualified.

## What the experiment distinguishes

The final factorial uses v=variance-first Normal arithmetic, b=Normal bandwidth
broadcast before variance/log evaluation, and l=left-associated dimension sums.
The von Mises reciprocal/broadcast order is held fixed. Each arm uses both dtypes
and all ordinary, seam, tiny-weight and far-tail fixtures. Bounds are unchanged:
128*2**-23*(1+abs(author)) and 1e-9*(1+abs(author)).

| Arm | Failed comparisons | Largest error / preserved bound |
|---|---:|---:|
| v0_b0_l0 | 5 | 6.876 |
| v0_b0_l1 | 4 | 6.876 |
| v0_b1_l0 | 4 | 6.876 |
| v0_b1_l1 | 4 | 6.876 |
| v1_b0_l0 | 6 | 12.430 |
| v1_b0_l1 | 5 | 9.302 |
| v1_b1_l0 | 4 | 12.430 |
| v1_b1_l1 | 4 | 12.430 |

All failures are float32. v0_b0_l0 reproduces the earlier five CPU failures;
source-style Normal ordering can increase the largest discrepancy. Matching
these finite expressions alone is insufficient. The connected consumer and
raw objective are distinct: passing the connected consumer does not waive a
raw-gradient failure or predict the full GPU result.

For the raw seam weight derivative, the independently checked derivative is
0.36366725 while the sum of absolute derivative terms is 537.02873: cancellation
ratio 1476.70. Source float32 gives 0.36351055 and TF gives 0.36336750. For the
seam bandwidth derivative the corresponding scale is 5,339,141.76 and derivative
598.85553: ratio 8915.58. Source gives 598.85382 and TF gives 598.84363.
Both implementations differ from the common-input float64 derivative.

Across the five preserved baseline CPU failures, observed source/TF differences
at the most offending coordinate are approximately 0.016–3.69 times
float32 epsilon times the sum of absolute derivative terms. This is explanatory
evidence, not a rigorous error bound. Transcendentals, mixture responsibilities,
products and gradient accumulation also contribute; no tolerance is changed.
A bound based only on the final gradient ignores this cancellation scale.

The derivation is in the plan. The independent check differentiates the I0
polynomial through dual Horner arithmetic and assembles mixture responsibilities
and chain-rule terms directly. Its largest error/bound is
0.000280306.
It checks the derivative identity and exact stopped-sample chain; it does not
supply an independent proof of the source density or the complete filter.

## Execution, repair and review

Attempt 01 covered only the connected consumer and recreated kernels for each
fixture. Its PASS label denotes completed execution only; it is ineligible for
the plan's complete-endpoint claim. Attempt 02 added all endpoints and reused
kernels, but the broadcast arm also changed circular reciprocal placement and
omitted the analytic phase. Its factorial attribution is not used. Attempt 03
isolates Normal broadcasting, restores the exact source Normal expression,
checks all endpoints, and adds the analytic derivation and source hashes.
Earlier outputs are preserved unchanged.

The final diagnostic passes its validity checks. Every one of 48 endpoint
functions traces exactly once across fixtures; the diagnostic uses stable
signatures, one CPU thread and no pfor or NumPy numerical path. GPU is
intentionally hidden and XLA is off. The no-CUDA startup message is expected.
The active training kernels were not modified. Current source files are checked
against the author's preserved hashes; final script, kernel, plan, and framework
source snapshots are saved. Exact commands and environment are in the manifests.
Attempt 03's complete process log is preserved beside its output directory.

Attempts consumed 78.080538 CPU seconds;
with 60 seconds conservatively reserved for routine checks, the ledger charge is
138.080538 CPU seconds against the existing allocation. The phase is closed:
three attempts used, zero GPU time, no automatic retry or promotion.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep float32 compatibility open | All eight arms fail old screen | Numerical promotion veto remains | Independent forward-error bound and downstream effect | Derive/calibrate before fresh fixture and optimizer validation | Author algorithm fails |
| Accept bounded cancellation explanation | 36/36 analytic checks; 928/928 float64 endpoints | No derivative/finite/trace validity failure | Component/log-density rounding and unseen regimes | Source-only fresh conditioning calibration | A new numerical tolerance is accepted |
| Continue full training | Diagnostic does not alter worker code or its contract | No campaign continuation veto | Training completion and heldout quality | Monitor source and TF development runs | Completed replication |

| Inference status | Finding |
|---|---|
| Hard veto screen | Float32 comparison failures remain; no invalid analytic result |
| Statistically supported ranking | None; fixed diagnostic fixtures are not stochastic qualification |
| Descriptive-only differences | Per-arm failure counts, error sizes and cancellation ratios |
| Default-readiness | Not established; CPU/non-XLA diagnostic only |
| Next evidence needed | Independently justified numerical bounds, fresh fixtures, actual optimizer checks, GPU/XLA and untouched trained-model qualification |

Post-run red team: the analytic check reuses evaluated component values, so it
isolates gradient algebra more strongly than density construction. An unseen
gradient term, invalid mixture regime, or failed fresh derivative/consumer check
would overturn the bounded interpretation. Neither this result nor one successful
development training pair meets the accepted NLL/RMSE confidence criteria.

Artifacts: [terminal summary](artifacts/controlled-smc-kdm-replication-20260928/kdm-circular-arithmetic-20261007-03/summary.json),
[conditioning](artifacts/controlled-smc-kdm-replication-20260928/kdm-circular-arithmetic-20261007-03/conditioning.json),
[aggregate and accounting](artifacts/controlled-smc-kdm-replication-20260928/kdm-circular-arithmetic-20261007-summary.json).
