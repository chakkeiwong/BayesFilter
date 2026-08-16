# SSL-LSTM q=20 seed-B mode diagnostic reset memo (2026-08-09)

## Current state

The mode-region and fixed-representative predictive diagnostic is complete.
Use the repaired `r2` artifacts, not the `r1` nearest-MAP canary, for any current
interpretation.

Primary result:

- all 4,000 retained states had positive `observation_weight.0.0`;
- pooled retained observation-weight range was `0.128007` to `1.205762`;
- no retained state entered the negative half-space containing the known
  negative MAP at `-0.587697`;
- positive and negative MAP fixed simulators were each distinguished from truth
  at `T=10,20,30,50,100`;
- every test had zero exceedances among 9,999 permutations and p-value `0.0001`;
- no joint decision or representative ranking was computed.

Therefore, low R-hat did not demonstrate coverage of the known negative region,
but missing that region alone does not explain the plug-in predictive failure:
the negative MAP fixed simulator displays the same qualitative discrepancy.

## Authority files

- Plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-plan-2026-08-09.md`
- Result:
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-result-2026-08-09.md`
- Structured summary:
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r2/summary.json`
- Region coverage:
  `docs/plans/artifacts/ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r2/occupancy.json`
- Runner:
  `docs/benchmarks/run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py`
- Focused tests:
  `tests/test_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic.py`

The `r1` canary is historical repair evidence only. Its raw-Euclidean
nearest-MAP assignment was falsified by its disagreement with the directly
observed weight sign and is ineligible for basin-occupancy interpretation.

## What is established

- Target-only multistart optimization found sign-separated stationary MAP
  representatives with nearly equal point log densities.
- Four retained chains passed their earlier within-region sequential R-hat/ESS
  screen.
- None of the 4,000 retained states entered the negative observation-weight
  half-space.
- Both fixed MAP simulator laws are rejected against truth at all five tested
  horizons under the predeclared energy tests.

## What is not established

- formal basin occupancy or cross-mode transition probability;
- integrated posterior mass of either mode;
- whether leapfrog intermediate states crossed the sign boundary;
- within-mode or mixture posterior-predictive validity;
- posterior correctness or incorrectness;
- whether finite-data posterior displacement, the filtering likelihood,
  predictive construction, or model specification causes the output shift;
- that recovering the negative mode cannot help at all.

## Next justified work

Do not repeat another fixed posterior-mean, median, or MAP plug-in comparison.
The next action depends on the research question:

1. For sampler coverage: create a new reviewed plan for chains initialized in
   both known regions, tune kernels per region without NUTS, and test whether a
   valid common posterior authority can combine them. This requires a mode-mass
   method; equal chain counts are not posterior weights.
2. For predictive validity: construct a reviewed within-mode and multimode
   posterior-predictive diagnostic. It must state how draws and mode weights are
   obtained; the present artifacts provide no valid negative-mode draw set or
   integrated weights.
3. For cause localization: test the target/filter/data pipeline at the true
   control and around both MAPs, separating finite-data posterior displacement
   from likelihood/filter approximation and predictive-simulator mismatch.

The smallest discriminating next step is cause localization before another
expensive HMC run: establish whether the exact target evaluated at truth versus
the two MAPs and the data-generating/predictive construction are mathematically
aligned. If they are aligned, then plan multimode-initialized posterior sampling.

