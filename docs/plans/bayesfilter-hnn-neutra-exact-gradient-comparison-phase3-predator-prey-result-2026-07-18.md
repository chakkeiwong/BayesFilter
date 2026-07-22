# HNN-NeuTra Exact-Gradient Comparison Phase 3 Result

Superseded for tuning and performance claims, 2026-07-18.  The fixed-grid
tuner used here was not the native BayesFilter dual-averaging tuner.  All
reported tuned timings, seconds/ESS, speed ratios, break-even values, and
performance decisions below are `UNSUPPORTED_PENDING_NATIVE_RETUNING`.  The
historical tables are retained for provenance only; see
`bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.

Decision: `PASS_PHASE3_TWO_OF_TWO_ONE_SEED_ACCURACY_AND_DESCRIPTIVE_PERFORMANCE`.

## Results

| Cell | Tuned HNN eps/L | Tuned exact eps/L | Matched exact/HNN speed ratio | HNN / exact sampling seconds | HNN / exact seconds per min bulk ESS |
| --- | --- | --- | ---: | ---: | ---: |
| PP-UKF | 0.4 / 10 | 0.2 / 10 | 25.435x | 93.930 / 2362.997 | 0.02051 / 0.33019 |
| PP-SGQF | 0.2 / 10 | 0.2 / 10 | 25.895x | 46.478 / 1136.869 | 0.006657 / 0.16320 |

The matched benchmark held chart, initial positions, seed, step size, `L=10`,
four chains, 500 transitions per chain, endpoint, dtype, GPU, and XLA fixed.
It alternated arm order over three synchronized warm repeats. Only the force
callable differed.

## Accuracy

Both arms in both cells passed:

- finite/status and exact energy-identity checks;
- retained maximum rank-normalized split/folded R-hat at most `1.01`;
- minimum bulk ESS at least `1000` and tail ESS at least `400`;
- generating-truth tail screen;
- physical 95% intervals and pooled-MCSE direct HNN-versus-exact comparison.

| Cell | HNN min truth tail | Exact min truth tail | Max direct `z_MC` | Second seed |
| --- | ---: | ---: | ---: | --- |
| PP-UKF | 0.2132 | 0.2017 | 1.7864 | no |
| PP-SGQF | 0.1927 | 0.1937 | 0.6975 | no |

Every direct interval pair overlapped. Neither truth nor direct agreement was
marginal under the predeclared rule.

## Preparation And Total Reuse Cost

| Cell | Supervision generation | HNN grid wall | HNN tuning | Exact tuning | HNN reuse total | Exact reuse total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PP-UKF | 141.153 s | 10.850 s | 164.801 s | 2942.987 s | 410.735 s | 5305.984 s |
| PP-SGQF | 44.919 s | 9.982 s | 86.391 s | 1400.827 s | 187.770 s | 2537.696 s |

The full independently tuned reuse campaign has immediate HNN break-even
because the saved exact-gradient tuning time already exceeds HNN
supervision/training overhead. HNN preparation itself is not free: it
amortizes after 200 matched transition batches for PP-UKF and 152 for
PP-SGQF. The common NeuTra chart-training cost remains
`not_reconstructed`; therefore no from-scratch total is claimed. The optimizer
alone took only seconds, but that is correctly labeled optimization-only and
is not the full HNN preparation cost.

## Interpretation

The hard-veto evidence supports both methods as viable in one seed on each
named deterministic filter posterior. The mechanism and tuned efficiency
differences are descriptively large and in the expected direction. No
statistically supported stochastic ranking is claimed from one seed.

The exact-gradient and HNN routes emitted three TensorFlow warnings about
casting complex diagnostic intermediates to float64 during post-chain
summaries. All validity-bearing target, force, energy, samples, convergence,
truth, and agreement outputs were finite real float64 values. The warnings did
not fire a hard veto, but their diagnostic source should be localized before a
publication-grade distributional-correlation claim.

## Decision Table

| Field | Status |
| --- | --- |
| Primary accuracy criterion | passed in both cells and both arms |
| Matched mechanism criterion | HNN faster in both cells |
| Tuned useful-sample criterion | HNN lower seconds/minimum bulk ESS in both cells |
| Hard veto status | clear |
| Main uncertainty | one seed/fixture; common chart-training cost unreconstructed |
| Next justified action | execute SIR-SGQF and STR-UKF independently |
| Not concluded | universal superiority, calibration, latent-model exactness, filter ranking, or default readiness |

## Inference Status

| Field | Status |
| --- | --- |
| Viable candidates | HNN and exact NeuTra-HMC in PP-UKF and PP-SGQF |
| Statistically supported ranking | none |
| Descriptive differences | matched time, tuning time, sampling time, ESS, acceptance, and speed ratios |
| Default readiness | not established |
| Next evidence | Phase 4 SIR/structural cells; second seeds only if their predeclared marginal rule fires |

Phase 4 was reviewed against the same two-arm baseline, fresh HNN preparation,
SIR truth contract, structural deterministic/no-noise contract, synchronized
timing, adaptive caps, and independent six-hour cell ceilings. No shared
continuation veto fired.
