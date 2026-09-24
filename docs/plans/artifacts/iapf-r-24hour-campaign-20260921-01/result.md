# Independent R 24-hour campaign result

Status: frozen campaign completed.

Charged R worker time: 107291.716/172800 seconds.
Full five-dimension study complete: True.

| Phase | d | Data seed | Arm | Complete | Practical | Heuristic | Literal |
|---|---:|---:|---|---:|---|---|---|
| phase01-controller-pending | 80 | 89500080 | qr | 8/8 | False | passed observed screen | False |
| phase01-controller-pending | 80 | 89500080 | short_qr | 8/8 | True | passed observed screen | False |
| phase02-controller-64 | 80 | 89600080 | qr | 64/64 | False | veto | False |
| phase02-controller-64 | 80 | 89600080 | short_qr | 64/64 | True | passed observed screen | True |
| phase02-controller-64 | 80 | 89700080 | qr | 64/64 | False | veto | False |
| phase02-controller-64 | 80 | 89700080 | short_qr | 64/64 | True | veto | True |
| phase03-first-study-1000 | 5 | 89800005 | qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 5 | 89800005 | short_qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 10 | 89800010 | qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 10 | 89800010 | short_qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 20 | 89800020 | qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 20 | 89800020 | short_qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 40 | 89800040 | qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 40 | 89800040 | short_qr | 1000/1000 | True | passed observed screen | False |
| phase03-first-study-1000 | 80 | 89800080 | qr | 1000/1000 | False | veto | False |
| phase03-first-study-1000 | 80 | 89800080 | short_qr | 1000/1000 | True | veto | True |

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve both frozen reconstructions and all failures | All-dimension screen: {'qr': False, 'short_qr': False} | Per-cell numerical and conditional heuristic results in linked reports | Unknown author settings; one fixed dataset per dimension; rare weights | Review completed study against paper tables and the remaining R/TF implementation gaps | Literal author replication, superiority, production or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical failures, missing records and heuristic vetoes are retained per cell. |
| Statistically supported ranking | No global ranking; only reported paired pointwise contrasts can support a conditional difference. |
| Descriptive-only differences | Unmatched-cost timings, extreme ratios and raw paper-table differences. |
| Default-readiness | No default changed. QR is not paper Equation15. |
| Next evidence needed | Source-setting resolution and multivariate full-filter R/TF parity, guided by complete study evidence. |

Reports preserve intervals, conditional errors, log-ratio underflow counts and exact provenance.
The strongest alternative explanation is data dependence or missed rare large weights. A new
independent-data replication could overturn a fixed-data comparison. Literal equivalence to
unavailable author settings/data remains the weakest part of any replication claim.

- phase01-controller-pending: phase01-controller-pending/analysis-v1/result.md
- phase02-controller-64: phase02-controller-64/analysis-v1/result.md
- phase03-first-study-1000: phase03-first-study-1000/analysis-v1/result.md
