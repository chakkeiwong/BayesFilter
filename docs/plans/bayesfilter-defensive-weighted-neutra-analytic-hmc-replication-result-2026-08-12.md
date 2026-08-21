# Frozen weighted NeuTra HMC seed replication result (2026-08-12)

## Verdict

All four predeclared independent HMC root seeds passed the frozen-kernel
replication contract on the analytic unequal-weight two-mode target. Every run
passed the canonical sequential R-hat/ESS checks, had no hard numerical or
target-status veto, visited both modes in every chain, and produced a 99%
minority-mass interval containing the analytic truth `0.2`.

The result supports this direct classification:

`four_root_seeds_statistically_compatible_on_one_frozen_transport_and_one_analytic_target`

It does not prove stationarity or distributional equality, and it does not
establish cross-training-seed, cross-target, SSL-LSTM, sampler-superiority, or
default-readiness claims.

## Frozen contract

- Analytic target: `0.8 N(mu_1,Sigma_1) + 0.2 N(mu_2,Sigma_2)` in four dimensions.
- Frozen checkpoint SHA-256:
  `af961871dcc3b626216d7500e695534f147ecfd9ba4fe0f9907f59018d40e8e5`.
- Frozen tuning artifact SHA-256:
  `6dfe2b8145040a18831a08032bfd61854189f2651e76c70842e59d4e4e12eb4f`.
- Kernel: fixed identity mass in `z`, `L=20`, epsilon
  `0.14091138276334744`.
- Controller: `bayesfilter_neutra_sequential_hmc_v1`.
- Four mode-aware chains; warm-up archived and excluded.
- TensorFlow/TFP GPU/XLA, float64, TF32 disabled, verified memory growth.
- Root seeds were fixed before outcomes: `(20260812, 91011)` through
  `(20260812, 91014)`.

## Results

| Root | Warm-up / chain | Retained / chain | Max R-hat | Min bulk ESS | Min tail ESS | Minority mass | 99% interval | Mean intervals | Covariance intervals |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,000 | 3,000 | 1.00847 | 5,952.4 | 1,072.0 | 0.18992 | [0.16934, 0.21049] | 4/4 | 16/16 |
| 1 | 2,500 | 2,000 | 1.00786 | 4,147.1 | 637.1 | 0.18213 | [0.15647, 0.20778] | 4/4 | 15/16 |
| 2 | 2,000 | 5,000 | 1.00947 | 8,889.4 | 1,663.8 | 0.19620 | [0.18037, 0.21203] | 4/4 | 16/16 |
| 3 | 2,000 | 2,000 | 1.00987 | 3,988.7 | 669.8 | 0.19575 | [0.17113, 0.22037] | 4/4 | 16/16 |

The four minority estimates have descriptive mean `0.190998`, sample standard
deviation `0.006571`, and range `[0.182125, 0.196200]`. These between-seed
numbers are descriptive only; no seed ranking or aggregate equality test was
predeclared or performed.

All 16 chains crossed between hard-assignment modes repeatedly. Transition
counts per chain were:

- root 0: `[726, 616, 561, 632]`;
- root 1: `[366, 408, 473, 396]`;
- root 2: `[1081, 1050, 1104, 991]`;
- root 3: `[376, 367, 443, 488]`.

Mode transitions are explanatory evidence against simple initialized-mode
trapping. They are not convergence or equality evidence by themselves.

## Numerical diagnostics

- Hard vetoes: none in any run.
- All required states, target values, target scores, and target-status rows were finite/valid.
- Native divergence status: not exposed by the TFP kernel. This is unknown, not zero divergences.
- Acceptance probabilities across archived chunks ranged approximately from
  `0.664` to `0.802`. Acceptance is explanatory only.
- Maximum finite absolute energy-error proxies were very large, from about
  `4.39e14` to `1.04e17` across roots. These correspond to extremely poor
  rejected proposals and are an explanatory numerical concern. They did not
  create nonfinite values or invalidate retained-state diagnostics under the
  declared controller, but they should be diagnosed before any broader method
  promotion.
- Root 1 missed one of 16 marginal covariance intervals. Marginal moment
  intervals were not combined into an uncalibrated joint veto.

## Provenance and failures

The first four launch attempts and two subsequent canary attempts stopped before
HMC transitions because provenance checks initially conflated the tuner-internal
transformed-adapter signature with the live sequential-adapter signature and
because CPU/GPU covariance evaluation differed by one float64 ULP. Investigation
established:

- the serious GPU target tensors exactly reproduce the v5 target signature;
- the deterministic 4,110-point value/responsibility/score comparison has zero
  observed difference;
- v5 tuner-internal transformed signature: `7d188d...`;
- v5 live sequential transformed signature: `6b4e5c...`;
- GPU/XLA canary v3 passed with all four chains moving and verified memory growth.

The failed launch/canary roots contain manifests only and no HMC samples. They
are preserved as engineering provenance, not scientific evidence.

The consolidator verified `492` result/archive receipts across all four runs.
The terminal campaign summary SHA-256 is
`33f5b59b831dea76818a88e9c161c19e02cb3a0e3405de3b036221494b0211e8`.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| All four frozen root-seed replications passed | Four of four pass sequential R-hat/ESS and analytic minority/mode gates | No hard veto; native divergence unavailable | One frozen training replication and one four-dimensional analytic target | Repeat HMC on freshly trained frozen transports selected by a neutral pre-HMC rule; diagnose energy-error tails | No equality/stationarity proof, sampler ranking, general NeuTra, SSL-LSTM, or default claim |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed in all four roots |
| Viable candidate | The fixed transport/kernel remains viable under four HMC roots |
| Statistically supported ranking | None; neither methods nor seeds were ranked |
| Descriptive-only differences | Minority estimates, R-hat/ESS, runtime, transitions, acceptance, and energy tails |
| Default readiness | Not assessed |
| Next evidence needed | Independent HMC roots on neutrally selected fresh frozen transports, followed by target-suite replication |

## Post-run red team

The strongest alternative explanation is that one favorably trained transport
plus mode-aware starts makes this target easy even though other frozen training
seeds could fail. A fresh transport selected without posterior peeking that
fails the same hard or analytic gate would overturn cross-transport robustness.
The weakest evidence is the single transport/target scope and the lack of native
divergence telemetry; the extreme finite energy-error tail also warrants focused
diagnosis.

## Verification

- Focused final test suite: `30 passed` (TensorFlow/gast deprecation warnings only).
- XLA canary: passed; XLA compilation explicitly logged.
- Serious artifacts: 559 files, approximately 11 MiB.
- Total retained draws: 48,000 across all roots and chains.
- Total archived warm-up draws: 34,000 across all roots and chains.
- Per-process wall time: 194 to 285 seconds; four processes ran concurrently,
  two per physical GPU.
