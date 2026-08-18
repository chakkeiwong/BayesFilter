# Frozen weighted NeuTra HMC replications

All four predeclared independent HMC root seeds passed on the same frozen
transport, analytic target, and fixed `L=20` kernel.

| Root | Retained / chain | Max R-hat | Min bulk ESS | Min tail ESS | Minority mass | 99% interval | Mode transitions in all chains |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3,000 | 1.00847 | 5,952.4 | 1,072.0 | 0.18992 | [0.16934, 0.21049] | yes |
| 1 | 2,000 | 1.00786 | 4,147.1 | 637.1 | 0.18213 | [0.15647, 0.20778] | yes |
| 2 | 5,000 | 1.00947 | 8,889.4 | 1,663.8 | 0.19620 | [0.18037, 0.21203] | yes |
| 3 | 2,000 | 1.00987 | 3,988.7 | 669.8 | 0.19575 | [0.17113, 0.22037] | yes |

No hard veto fired. Native divergence telemetry was not exposed and must not be
interpreted as zero. Extreme but finite energy-error proxies remain an
explanatory concern. These runs condition on one frozen training replication
and one analytic target; no general NeuTra, SSL-LSTM, equality, stationarity,
sampler-ranking, or default-readiness claim follows.

Machine-readable authority: `summary.json`, SHA-256
`33f5b59b831dea76818a88e9c161c19e02cb3a0e3405de3b036221494b0211e8`.
