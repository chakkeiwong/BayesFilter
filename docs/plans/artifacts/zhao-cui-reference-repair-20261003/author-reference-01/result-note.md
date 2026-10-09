# Zhao--Cui source reference execution

This is a bounded Octave mechanics check of the pinned author solver with recorded local adaptations.
The corrected statistic averages importance weights before taking the logarithm. It remains sensitive to proposal support and Monte Carlo error.

| Model | Status | Corrected logmeanexp | Legacy mean log weight | ESS | Finite fraction |
|---|---|---:|---:|---:|---:|
| pp | failed | None | None | None | None |
| sir_austria | failed | None | None | None | None |

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Preserve source reference mechanics | See numerical status and parity in results.json | Nonfinite/provenance/alignment failures block numerical admission | Tiny rank and sample size; support and Monte Carlo error | Convergence and independent-reference validation before oracle use | No score, accuracy, ranking, production, or HMC certification |

| Inference item | Status |
|---|---|
| Hard veto screen | Per-case finite-weight and source-provenance checks in results.json |
| Statistically supported ranking | None; no method comparison |
| Descriptive differences | Corrected and legacy values, ESS, TT normalizer, runtime |
| Default readiness | Not evaluated |
| Next evidence | Adequate TT/sample convergence, uncertainty, support checks and independent likelihood comparison |

Post-run red team: a finite corrected value can be very inaccurate when a single trajectory dominates. Model parity tests do not test TT posterior quality. The source PP bounded proposal may miss target mass. Multi-setting convergence with independent reference agreement could overturn these accuracy concerns.

No analytical observed-data score is implemented by this bridge. Raw log weights, path samples, resolved call chain and exact commands are retained.

CPU wall seconds: 0.537. Mode: author. Plan: docs/plans/zhao-cui-reference-repair-20261003.md.
