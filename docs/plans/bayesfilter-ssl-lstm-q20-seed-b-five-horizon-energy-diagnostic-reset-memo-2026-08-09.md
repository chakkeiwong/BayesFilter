# Reset memo: seed-B five-horizon energy diagnostics (2026-08-09)

## State

The requested campaign is complete. No process is running and no recovery is
needed. The terminal summary is:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-five-horizon-energy-diagnostic-2026-08-09/r1/summary.json`

Status: `FIVE_DIAGNOSTICS_COMPLETED`.

## Result

All five separate equality tests rejected at the 1% level:

| T | Energy | p-value | Status |
|---:|---:|---:|---|
| 10 | `0.193903` | `0.0001` | distinguished |
| 20 | `0.293548` | `0.0001` | distinguished |
| 30 | `0.331538` | `0.0001` | distinguished |
| 50 | `0.429572` | `0.0001` | distinguished |
| 100 | `0.586977` | `0.0001` | distinguished |

Each row had zero exceedances among 9,999 balanced label permutations. The
p-value `0.0001` is the minimum possible with the plus-one correction and does
not mean the exact permutation p-value is known to equal `0.0001`.

The posterior-mean simulator has an average raw output shift of approximately
`+0.27` relative to the true-control simulator, which plausibly drives much of
the separation. Energy does not identify the unique cause.

## Interpretation boundary

- Correct: the two fixed simulators are statistically distinguishable for each
  of the five tested complete-path distributions at `n=1000` and `alpha=0.01`.
- Unsupported: a combined p-value or joint five-horizon test; none was computed.
- Unsupported: practical inequivalence, posterior invalidity, NeuTra failure,
  model inadequacy, or a claim about all possible horizons.
- Passing all five would not have proved equality. In fact, none passed under
  the equality-test decision rule.

Under independent tests, `1-0.99^5=0.0490099501` is the chance of at least one
false rejection if all five nulls hold. It is not a combined p-value. The five
diagnostic paths and permutation seeds were disjoint, but the hypotheses remain
scientifically related.

## Changed files

- `bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py`: default-preserving
  arbitrary positive forecast horizon.
- `bayesfilter/testing/two_sample_energy_tf.py`: diagnostic-only TensorFlow/XLA
  whole-path energy permutation test.
- `docs/benchmarks/run_ssl_lstm_q20_seed_b_five_horizon_energy_diagnostic_2026_08_09.py`:
  campaign runner.
- `tests/test_ssl_lstm_complexity_predictive_tf.py`: arbitrary-horizon tests.
- `tests/test_two_sample_energy_tf.py`: statistic mechanics tests.
- `tests/test_ssl_lstm_q20_seed_b_five_horizon_energy_diagnostic.py`: campaign
  contract tests.
- The plan, result, reset memo, and versioned structured artifact root.

## Verification

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_ssl_lstm_complexity_predictive_tf.py \
  tests/test_two_sample_energy_tf.py \
  tests/test_ssl_lstm_q20_seed_b_five_horizon_energy_diagnostic.py
```

Expected: 20 tests pass.

## Next justified question

No more equality testing is necessary to show these two fixed simulators differ
at the tested horizons. If the research question is whether the difference is
scientifically acceptable, the next plan must define a practical criterion. If
the question is why they differ, decompose location, scale, marginal shape, and
temporal dependence using descriptive or predeclared diagnostic components.

