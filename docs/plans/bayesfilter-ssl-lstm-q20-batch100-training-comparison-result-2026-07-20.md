# SSL-LSTM q=20 Batch-100 Training Comparison Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-batch100-training-comparison-plan-2026-07-20.md`  
Status: `COMPLETED_TRAINING_DIAGNOSTIC_VETOED`

## Run

The valid run is recorded under:
`docs/plans/artifacts/ssl-lstm-q20-batch100-training-comparison-2026-07-20/`.

| Field | Value |
| --- | --- |
| q | 20 |
| stream | seed-a |
| batch size | 100 |
| validation batch | 64, same seed and rows as batch-480 comparator |
| parameters | learning rate `4e-4`, initialization scale `0.01`, clip norm `10` |
| architecture | three-stage dense IAF, hidden `(32,32)` |
| maximum steps | 2,000 |
| validation cadence | every 250 optimizer steps |
| selected physical GPU | 1 |
| parent | XLA enabled, TF32 enabled, float64 tensors |
| workers | 16 CPU-hidden value/score workers |
| charged time | 1,183.86 seconds |
| parent high-water RSS | 1.11 GB |
| declared cap | 7,200 seconds |

## Validation History

| Step | Training draws | Mean heldout loss | Saturation | Action |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 79.75985549 | 0.000000 | initialize best |
| 250 | 25,000 | 43.92945791 | 0.002604 | improved |
| 500 | 50,000 | 42.51702785 | 0.041667 | improved |
| 750 | 75,000 | 42.27151301 | 0.075521 | stop: saturation cap |

The batch-480 comparator's best checkpoint was step 500 with heldout mean loss
`42.16940131` after `240,000` training draws. Batch 100 had a best observed loss
of `42.27151301` at step 750 before its saturation veto. This is a training
diagnostic only, not evidence that the batch-100 model has a different
out-of-sample predictive law, because no posterior-predictive forecast
comparison was run.

## What Can Be Inferred About Epochs

This harness samples a fresh stateless training batch at every optimizer step;
there is no finite dataset and hence no native epoch.  If one calls one
batch-480 draw block an ``equivalent epoch``, then:

```text
batch-100 equivalent epochs at step n = (100*n)/480 = n/4.8.
```

The run reached `750` batch-100 updates, or `75,000` draws, equal to `156.25`
batch-480 draw blocks. This exposure conversion is a cost diagnostic only;
there is no scientific requirement that training losses or draw blocks match.
The saturation veto prevents extrapolating either a step count or a predictive
result from the three observed points.

## Predictive Equivalence Status

No out-of-sample posterior-predictive comparison was performed. A valid future
comparison requires both batch-size arms to pass transport and four-chain
retained-HMC validity, then uses held-out forecast observations or disjoint
forecast innovations at horizons 1--10. The estimand should be the
batch-100-minus-batch-480 vector of predictive means and log variances (or the
predeclared proper-score loss), with dependence-aware simultaneous confidence
regions and a practical-equivalence margin calibrated before seeing arm
outcomes. The existing Phase 8 margin/sample-size ladder was underpowered and
did not freeze such a design, so this artifact supports no formal
predictive-equivalence, material-difference, posterior-correctness, or HMC
claim.

## Validity And Interpretation

- Launch, memory-growth, finite-value, worker-visibility, checkpoint, support,
  and round-trip checks passed.
- The stream was vetoed solely because saturation exceeded the existing `0.05`
  cap at step 750; learning-rate repair was not reached.
- This weakens the fixed-smoke batch-100 training candidate under the current
  controller. It does not show that batch 100 is intrinsically invalid, that
  q=20 geometry is invalid, that NeuTra fails, or that its predictive law
  differs from batch 480.
- One seed and one batch size do not establish an optimal batch size, an epoch
  count, convergence, posterior correctness, HMC readiness, or default
  readiness.

## Next Discriminating Action

If a batch-100 route remains desirable, first run a separately authorized q=20
target-specific tuning or lower-learning-rate repair that prospectively states
whether saturation is a repair trigger or a hard stop. Only after both arms are
admitted and the predictive sample-size/margin design is powered should we run
the out-of-sample proper-score/equivalence comparison. Do not infer a
predictive difference or equivalence from this stopped training trajectory.
