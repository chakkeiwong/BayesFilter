# SSL-LSTM NeuTra Phase 8 Target-Pilot Native Review

Date: 2026-07-17

Verdict: `AGREE_TARGET_PILOT_ONLY`

Reviewed paths:

- `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-plan-2026-07-17.md`
- `docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py`
- `bayesfilter/inference/predictive_equivalence.py`
- focused pilot and predictive-equivalence tests

## Findings And Repairs

1. The first split wording called confirmation values unparsed. A serialized
   TensorFlow tensor is indivisible, so selecting prefix `0..63` deserializes
   the entire first `256`-draw shard. The contract now states that exactly:
   only `0..63` are selected/evaluated, `64..255` are not used in any
   computation, and segment 1 remains hash-only.
2. A full-path max-statistic bootstrap would materialize unnecessarily large
   tensors at Phase 9 scale. The numerical surface now computes exact pooled
   mean/log-sample-variance estimates and per-retained-draw influence vectors,
   averaging the two forecast replications inside each draw cluster. Existing
   chain-batch long-run covariance then supplies uncertainty without false
   arm pairing or full resampled paths.
3. Bandwidth language can confuse squared distances with Gaussian-kernel
   bandwidths. A new TensorFlow/XLA primitive freezes the median positive
   off-diagonal Euclidean distance, the same unit consumed by the RBF kernel,
   and fails closed on a duplicate-degenerate cloud.

No unresolved material finding remains for the bounded target pilot.

## Audit

| Risk | Disposition |
| --- | --- |
| Wrong baseline | Synthetic controls remain power authority; the excluded pooled target prefix supplies only target-specific scales/bands |
| Proxy promotion | Pilot scales, median distance, covariance, ridge, and timing cannot freeze power or decide equivalence |
| Leakage | Fixed `0..63` prefix is permanently excluded; no arm difference is computed; suffix forecasts remain unopened |
| Hidden assumption | The A2 two-replication contract is retained rather than silently enlarged; exact shard-deserialization scope is disclosed |
| Feasibility | 256 mapped points per chart match a static XLA program; warm canary timing supports the 1,200-second cap |
| Artifact coverage | Receipt binds all shard/sidecar hashes, selection indices, mappings, banks, pooled calibration quantities, devices, traces, timing, and nonclaims |
| Stop conditions | Source/hash/split/finite/covariance/seed/distance/ridge/GPU/XLA/serialization/resource vetoes are executable |

## Checks

- Pilot plus predictive-equivalence suite: `63 passed`.
- Predictive-equivalence suite after both new primitives: `58 passed`.
- Python compilation passed.
- `git diff --check` passed for all pilot/numerical paths.
- Pilot receipt path was unopened before launch.

The review authorizes only the one frozen target-pilot invocation. Controlled
null/power calibration and Phase 9 remain closed.

Post-failure repair addendum: the original pilot failed on six G projection
statuses. A terminal-only diagnostic localized the defect to eigendecomposition
inside the per-point XLA map. A row/column repair was tested, falsified, and
reverted. Separate raw-decomposition evidence showed batched XLA `eigh`
reconstructed the same covariances under unchanged gates. The staged batched
audit then passed all exact G/H prefix terminal points, with both compiled
stages tracing once. Target-pilot repair 01 preserves every pilot input and
statistical choice and writes a fresh receipt. Verdict:
`AGREE_TARGET_PILOT_REPAIR_01_ONLY`.
