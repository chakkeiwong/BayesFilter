# Reset memo: q=20 seed-B predictive-equivalence repair (2026-08-08)

## Current state

The repaired output-law comparison campaign is stopped at calibration. There is
no infrastructure blocker and no need to recover a process. The terminal active
artifact is:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-repair-2026-08-08/r3/nominate.json`

Its status is `NOMINATION_FAILED`. No `validate.json`, `material.json`, or
`audit.json` exists in any repair attempt. The seed-B posterior-mean candidate
was therefore not evaluated by the repaired formal harness.

## What was completed

- Implemented the repaired CPU/XLA runner at
  `docs/benchmarks/run_ssl_lstm_q20_seed_b_predictive_equivalence_repair_2026_08_08.py`.
- Added five focused enforcement tests at
  `tests/test_ssl_lstm_q20_seed_b_predictive_equivalence_repair.py`.
- Froze and audited the plan at
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-predictive-equivalence-repair-plan-2026-08-08.md`.
- Executed coherent `r2` and `r3` canary, scale, and nomination stages.
- Wrote the terminal result at
  `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-predictive-equivalence-repair-result-2026-08-08.md`.

## Terminal verdict

The engineering repairs are valid under the focused checks. The calibration
design is not admitted.

`r2` at 1,024 draws/lane was broadly underpowered. `r3` at 16,384 draws/lane
repaired feature intervals, but the MMD tolerance could not simultaneously
admit the negligible `+0.05` mean family and reject the shape-only skew family:

- tolerance `0.0015`: negligible mean `7/20 PASS`, shape `17/20 MATERIAL`;
- tolerance `0.0020`: negligible mean `16/20 PASS`, shape `7/20 MATERIAL`.

All `r3` decisions had zero hard vetoes. This is a decision-design separation
failure, not a target, XLA, covariance, forecast, or serialization failure.

## Do not do next

- Do not run `--mode validate`, `--mode material`, or `--mode audit`; their
  required passed receipts do not exist and the runner will fail closed.
- Do not infer that the posterior-mean candidate passed or failed.
- Do not tune a tolerance using the unopened material comparison.
- Do not simply add more draws without a prospective power argument; 16,384
  draws/lane already failed the operating gate.
- Do not propagate all 4,000 HMC parameters as a mixture. The user-requested
  target remains one posterior-mean physical parameter vector.

## Next justified work

Create a new prospective calibration plan before running more seeds. The plan
should preserve the user’s scientific target but revisit the shape/dependence
discriminator. Candidate directions include a more efficient iid two-sample
path statistic with a valid uncertainty procedure, or a separately calibrated
shape/dependence feature family. It must state whether raw `+0.05` remains the
negligible q=20 mean margin and whether skew coefficient `0.35` remains the
minimum material shape alternative; those are policy hypotheses, not facts.

Any new design must use fresh calibration seeds, nominate on at least the same
six operating families or justify a change, validate on a disjoint 60-replication
set, and keep material seeds closed until validation passes.

## Verification command

```bash
pytest -q tests/test_ssl_lstm_q20_seed_b_predictive_equivalence_repair.py
```

Expected: `5 passed`.

