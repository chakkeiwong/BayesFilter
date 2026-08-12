# SSL-LSTM q=20 seed-B five-horizon energy diagnostic result (2026-08-09)

## Outcome

The five requested whole-path equality diagnostics completed successfully. At
each of `T = 10, 20, 30, 50, 100`, 1,000 independent complete paths were drawn
from the true-control simulator and 1,000 from the seed-B posterior-mean
simulator. Every T-specific energy permutation test rejected equality at the
user-fixed 1% level.

For every horizon, zero of 9,999 balanced-label permutation statistics equaled
or exceeded the observed statistic. With the plus-one Monte Carlo correction,
each reported p-value is `0.0001`, the minimum attainable value under this
permutation budget. This does not establish that an exact permutation p-value
is zero or smaller than `0.0001` by a known amount.

No joint test, combined p-value, multiplicity adjustment, or all-horizon pass
criterion was computed.

## Results

| Horizon T | Energy statistic | Permutation q99 | Exceedances / 9,999 | Monte Carlo p-value | T-specific decision |
|---:|---:|---:|---:|---:|---|
| 10 | `0.193903` | `0.012859` | 0 | `0.0001` | `DISTINGUISHED_AT_1_PERCENT` |
| 20 | `0.293548` | `0.016144` | 0 | `0.0001` | `DISTINGUISHED_AT_1_PERCENT` |
| 30 | `0.331538` | `0.018586` | 0 | `0.0001` | `DISTINGUISHED_AT_1_PERCENT` |
| 50 | `0.429572` | `0.022365` | 0 | `0.0001` | `DISTINGUISHED_AT_1_PERCENT` |
| 100 | `0.586977` | `0.029876` | 0 | `0.0001` | `DISTINGUISHED_AT_1_PERCENT` |

The point estimates become larger with T, but the five rows use different path
dimensions and independent samples. This table does not support a statistical
ranking of effect magnitude across horizons.

## Claimed and computed targets

| Item | Verdict |
|---|---|
| Claimed target | At each fixed T, equality of the two probability laws on complete raw T-step paths. |
| Quantity computed | Biased empirical whole-path energy distance and a 9,999-draw balanced-label Monte Carlo permutation p-value. |
| Relation | The statistic is a valid whole-path two-sample equality diagnostic under the iid-path and exchangeability null. |
| Supported verdict | The equality null is rejected separately at each of the five tested horizons at `alpha=0.01`. |
| Not proved | A quantified practical difference, posterior invalidity, parameter identification failure, inequality at every possible finite horizon, or inequality of an infinite-path stochastic process law. |

## Descriptive diagnostics

The posterior-mean arm had a persistent higher raw output mean:

| T | True-arm overall mean | Posterior-mean-arm overall mean | Average horizon mean difference | Maximum absolute horizon mean difference |
|---:|---:|---:|---:|---:|
| 10 | `0.00399` | `0.27892` | `0.27493` | `0.32936` |
| 20 | `-0.01110` | `0.27154` | `0.28264` | `0.36341` |
| 30 | `-0.00442` | `0.26723` | `0.27164` | `0.35149` |
| 50 | `0.00263` | `0.27165` | `0.26902` | `0.37804` |
| 100 | `0.00244` | `0.26832` | `0.26588` | `0.35324` |

These mean rows explain an important source of distinguishability but are not
separate tests and do not decompose the energy rejection uniquely. Energy
distance also responds to variance, marginal shape, and temporal dependence.

## Multiplicity interpretation

There were five tests, not four. Each had per-test significance level `0.01`.
If all five equality nulls held and the tests were independent, the probability
of at least one false rejection would be

`1 - 0.99^5 = 0.0490099501`.

That number is a familywise false-rejection probability under an independence
assumption, not a combined p-value. The run used disjoint simulator and
permutation seeds, making the Monte Carlo rows independent conditional on the
two fixed black boxes. The scientific hypotheses are still related across T,
and no joint interpretation was requested or made. In this run all five tests
rejected individually, so the earlier “pass all five” thought experiment does
not arise.

## Engineering work and verification

The prior q=20 simulator hard-coded horizon 10. It now accepts a positive
integer `horizon` argument while preserving `horizon=10` as the default and
binding the horizon in the XLA program cache, input shapes, output receipt, and
construction signature.

A diagnostic-only TensorFlow implementation was added for:

- exact pooled pairwise Euclidean distances on complete paths;
- the equal-arm energy V-statistic;
- balanced stateless label permutations in bounded batches;
- the plus-one Monte Carlo p-value; and
- finite, symmetry, diagonal, shape, replay, and non-negativity checks.

Focused verification passed:

- 11 q-general predictive tests, including arbitrary `T=20` XLA replay;
- 6 energy tests, including direct-formula identity, replay, eager/XLA parity,
  large-shift detection, and invalid-geometry vetoes; and
- 3 runner contract tests covering five horizons, `n=1000`, 9,999
  permutations, strict `<0.01`, disjoint seeds, and no joint decision.

The pre-run true-versus-true canary at `T=20`, `n=32`, and 999 permutations
completed with energy `0.403044` and p-value `0.167`. This is a mechanics and
runtime check only. One null realization does not calibrate Type-I error.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Horizon API admission for this diagnostic | Focused XLA replay and legacy-default tests passed | No shape, replay, or finite veto | Broader callers were not exhaustively retested | Preserve default `T=10`; use explicit horizon in diagnostics | General API/default promotion beyond this extension |
| Energy mechanics | Direct formula, replay, and XLA parity passed | No numerical or permutation-geometry veto | Finite Monte Carlo permutation resolution | Use structured receipts as diagnostic evidence | Universal power or optimality of energy distance |
| T=10 equality | `p=0.0001 < 0.01` | None | Monte Carlo p-value has floor `0.0001` | Classify as distinguished | Practical importance or cause |
| T=20 equality | `p=0.0001 < 0.01` | None | Same | Classify as distinguished | Same |
| T=30 equality | `p=0.0001 < 0.01` | None | Same | Classify as distinguished | Same |
| T=50 equality | `p=0.0001 < 0.01` | None | Same | Classify as distinguished | Same |
| T=100 equality | `p=0.0001 < 0.01` | None | Same | Classify as distinguished | Same |
| Joint/all-horizon conclusion | Not computed | N/A | Hypotheses are related and only five finite horizons were tested | None under this diagnostic request | Combined p-value, familywise decision, or infinite-process claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for all five horizons: finite paths/statistics, valid target/source identities, correct shapes, XLA, CPU device, balanced permutations, and valid receipts. |
| Statistically supported conclusion | Each tested T-specific equality null is rejected at the predeclared 1% level. |
| Descriptive-only differences | Energy magnitudes across T, output mean/variance differences, permutation quantiles, and runtimes. |
| Statistically supported ranking | None across horizons; dimensions and independent samples differ. |
| Default readiness | Not established. This was a CPU-only diagnostic exception. |
| Next evidence needed | If practical rather than exact equality matters, specify an acceptable scientific discrepancy or determine which output components cause the rejection. No additional equality test is needed to show current distinguishability at these T and n. |

## Research ledgers

Engineering correctness: the new horizon argument and energy implementation
passed focused tests and all serious-run invariants. A pre-run unit test exposed
roundoff on Gram-matrix diagonal distances; the implementation now sets the
mathematically exact zero diagonal after symmetrization. This repair occurred
before the canary or campaign seeds were opened.

Numerical/statistical validity: paths are iid across simulator calls, complete
paths are the observation units, labels are balanced, arm sizes are equal, and
the Monte Carlo p-value uses the plus-one correction. There were no hard vetoes.

Scientific interpretation: the two fixed black boxes are readily statistically
distinguishable at every tested finite horizon. This is evidence against exact
equality of those five finite-dimensional output laws. It is not evidence that
the posterior mean is scientifically unacceptable without a practical effect
criterion, nor does it identify which posterior or training mechanism caused
the difference.

## Post-run red team

Strongest alternative explanation: energy distance may be driven predominantly
by the persistent `~0.27` location shift rather than a broader dynamic mismatch.
That does not invalidate the equality rejection because location is part of the
raw path law, but it limits any claim about shape or temporal dependence.

What would overturn the equality verdict: a replay with independently generated
paths and permutations that yields p-values above 1% could show stochastic
instability, but the current separation between observed statistics and even
the permutation q99 values is large. A parity implementation of the same
statistic yielding materially different values would invalidate the harness;
the direct-formula and eager/XLA checks currently argue against that.

Weakest evidence: no empirical power curve or repeated-null Type-I calibration
was requested. Consequently, non-rejection would have been weak diagnostic
evidence. The observed strong rejections are still valid under the standard
permutation-test assumptions, but this run does not quantify the smallest
detectable or scientifically meaningful difference.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`, dirty concurrent worktree |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| Device | CPU only; GPU hidden before TensorFlow import; 8 TensorFlow intra-op threads |
| CPU exception reason | Both visible GPUs were 95-98% utilized by concurrent work during planning |
| XLA | Enabled for forecast, distance, and permutation statistic kernels |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Parameter source | Plug-in artifact SHA-256 `72ba9c7034e36f26e76d0d6542c3aa0ab6699e4d21fe0f727ca5dea275663f09` |
| Seeds | Disjoint per-arm and permutation seeds recorded in every T receipt |
| Campaign wall time | `166.6808 s` |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-seed-b-five-horizon-energy-diagnostic-2026-08-09/r1/` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-five-horizon-energy-diagnostic-plan-2026-08-09.md` |
| Result | This file |

Structured JSON receipt SHA-256 values:

- `canary.json`: `8a2da9300346bb91be40ee104e94fb43fc2ba2d3a93b30ad5cea34ee7995663a`
- `t010.json`: `402198f0b48de23d658fbf4e432d80af15a100d79e610b9d7eac6dbfbf4e4951`
- `t020.json`: `ae8efcded9aa761b801d0e72bfc0bdd782c1da117bbef15ce1183e3a243f0282`
- `t030.json`: `74ee0f81a1904cda25dad4ce04d0e18e4b67f418dc9f3cffde003d869f13b1bc`
- `t050.json`: `11f789f16fb5829ca69ba42a427ae991adfa69d0617da97a441d86d5d6fa3de0`
- `t100.json`: `8c59648057a5cd5c36d616cfc3a8663248f0dc24db25f7ec13ef4ee12033d528`
- `summary.json`: `598557797e805d85cb631adedf63b190773c8b834f350b13712b5e144b852f02`

