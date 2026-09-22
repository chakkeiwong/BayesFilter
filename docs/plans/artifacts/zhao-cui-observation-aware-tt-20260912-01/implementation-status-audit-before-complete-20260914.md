# Observation-aware Zhao--Cui TT implementation status

Audit date: 2026-09-13

The LaTeX/PDF note contains the SGQF discussion and the mathematical recursive
proposal.  The original Python master driver remains a Step-0/test smoke
driver.  A second driver now assembles the complete bounded scalar diagnostic
lifecycle, while the paper-scale source-faithful route remains open.

| Required stage | Current status | Evidence |
|---|---|---|
| SGQF guide construction and observation-response smoke | Implemented for a tiny actual-SV fixture | `run_observation_aware_tt_repair_master.py::_step0` |
| Existing actual-SV TT/moment-teacher checks | Implemented as subprocess tests | `run_observation_aware_tt_repair_master.py::_tt_tests` |
| Retained posterior bank carrying `y_{1:t-1}` through time | Implemented in bounded scalar diagnostic | `docs/benchmarks/run_observation_aware_tt_repair_full_master.py` loop over `tt_x, tt_w` |
| Observation-conditioned guide wired into TT coordinates/row law | Implemented as SGQF guide and defensive mixture | `_guide`, `_conditional_grid`, and per-step call chain |
| Exact pulled-back target, TT fit, KR conditional proposal | Implemented as fixed-rank TT plus piecewise-linear CDF inverse | `_fit_step_tt`, `_tt_proposal_step`; CDF inverse is explicitly extension/invention |
| Proposal density, determinant, and exact importance correction in a recursive SV run | Exercised for four steps | `exact_correction_*` fields in `run-20/result.json`; quadratic CDF-inverse residual is also recorded |
| Identity-ancestry particle update and resampling | Deterministic systematic resampling is implemented at an ESS threshold; the run records whether it fired | `_systematic_resample`; `resampling_policy` in the result |
| Multi-step horizon, independent reference, analytical score, and comparison ladder | Horizon and prior-SIR comparator implemented; dense reference and analytical score remain open | `tt_steps`, `prior_sir_steps`, and result note |
| Promotion/default-readiness conclusion | Correctly withheld | `result.json` records explicit nonclaims |

Therefore the current status is:

`bounded scalar observation-aware TT lifecycle executes and passes mechanical
validity screens; source-faithful paper-scale Zhao--Cui repair is not yet
implemented or validated.`

The note's complete algorithm is a mathematical specification and staged plan,
not evidence that every stage has a production call chain.
