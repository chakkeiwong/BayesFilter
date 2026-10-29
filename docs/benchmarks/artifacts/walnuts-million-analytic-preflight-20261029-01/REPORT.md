# Million-iteration funnel replication results

Pinned author algorithm/protocol; four independent method processes. Concurrent wall times include contention.
Python reference evaluator: normalized analytic funnel_reference.target(11). TF uses the equivalent analytic formula without its constant normalization.
The original and corrected Python arms remain separate. Newly selected h/delta need not equal the paper’s rounded values.

| Method | R-hat max | Bulk ESS min | Tail ESS min | Screen | Sampling seconds |
|---|---:|---:|---:|---|---:|
| author-original | N/A | N/A | N/A | mechanics smoke only | N/A |
| author-corrected | N/A | N/A | N/A | mechanics smoke only | N/A |
| tf-cpu | N/A | N/A | N/A | mechanics smoke only | N/A |
| tf-gpu | N/A | N/A | N/A | mechanics smoke only | N/A |

## Decision table

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Assess compatibility | Complete draws and posterior screens above | Missing/invalid artifacts or failed screens | Batch MCSE and initialization sensitivity | Inspect comparison.json intervals; investigate differences | Formal equivalence or universal sampler validity |

## Inference status

| Item | Finding |
|---|---|
| Hard veto screen | Per-arm status above; missing arms: [] |
| Statistically supported ranking | None; concurrent single-run timing is not a speed ranking |
| Descriptive differences | Runtime, ESS, tails, histograms and failed-screen intervals |
| Default readiness | No change |
| Next evidence | Independent tuning/seed replications and isolated timing if a ranking is needed |

## Result review

Strongest alternative explanations are finite-chain mixing, fixed tuning shared from one author run, and original-source weight error. A passed finite-sample screen is not proof of stationarity. Inspect 50/100-block MCSE sensitivity and simultaneous intervals before interpreting reference agreement. The extreme v=-30 test and exact figure-level replication are separate.
