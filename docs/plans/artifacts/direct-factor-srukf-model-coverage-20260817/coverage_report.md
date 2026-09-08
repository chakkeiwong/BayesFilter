# Direct-Factor SR-UKF Model Coverage Report

Campaign status: `EXECUTED_WITH_EXPLICIT_CLASSIFICATION_BOUNDARIES`.

Inventory rows: 24; executed rows: 6.
Status counts: adapter_required=4, blocked=6, eligible_score=5, eligible_value_only=1, historical_only=1, not_applicable_contract=5, owner_excluded=2.

Executed score rows are the three existing direct-factor fixtures (A/B/C) plus the certified PP-UKF and STR-UKF one-time factor adapters, with eager/XLA attempts, historical principal-root comparison, and fixed-program score evidence. The structural AR1 row was executed through the rectangular direct-stack SVD support route and is value-only.

Common V2 and NeuTra rows are all classified. Models whose source route is principal-root UKF, SGQF, multiplicative SV, domain-constrained SIR, DPF/LEDH, or an un-certified covariance adapter are not silently promoted to direct-factor evidence.

Singular robustness includes QR scales 1 through 1e-15, exact rank-zero/rank-one stacks, repeated singular values, reconstruction residuals, and explicit value-only branch metadata.

Nonclaims: no claim of exact nonlinear Bayesian inference, no score through rank/support changes, no HMC readiness, and no repository-wide claim that inapplicable contracts were tested as SR-UKF.
