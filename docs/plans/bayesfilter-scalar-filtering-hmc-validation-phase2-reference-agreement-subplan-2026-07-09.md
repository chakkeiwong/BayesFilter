# Phase 2 Subplan: Local Quadratic Reference Agreement

Date: 2026-07-09
Status: `REVISED_AFTER_CODEX_SUBSTITUTE_REVIEW_ROUND_1`

## Phase Objective

Construct an auditable local quadratic Gaussian reference in the HMC execution
coordinate `u` from the accepted 2026-07-08 geometry/mass handoff, then compare
the Phase 1R retained HMC marginal moments against that local reference using a
predeclared criterion.

This phase is a local-geometry agreement check, not an exact posterior oracle.
It may pass or fail a local quadratic-reference agreement screen, but it cannot
establish posterior correctness, HMC convergence, or HMC readiness.

## Entry Conditions

- Phase 1 result exists and records the original acceptance-boundary failure.
- Phase 1R result exists and passes the longer same-kernel finite/acceptance
  repair screen.
- Phase 1R JSON artifact exists:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json`
- Current scalar target coordinate and mass mapping are recorded:
  `free = center + scale * z` and `z = u @ chol(M_z).T`.
- Native divergence remains unavailable/not-exposed and must not be treated as
  zero divergence.

## Required Artifacts

- Phase 2 harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py`
- Local-reference agreement JSON and Markdown:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2_local_quadratic_reference.log`
- Phase 2 result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-result-2026-07-09.md`
- Refreshed Phase 3 subplan only if Phase 2 passes.

## Required Checks, Tests, And Reviews

- Run `git diff --check` before and after edits.
- Run focused tests:
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2_reference.py`
  - `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase1r.py`
- Review this subplan with Claude or documented Codex fallback before runtime.
- Create the quiet log directory before redirected runtime:

```bash
mkdir -p docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09
```

- Run the comparison command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 180 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2_local_quadratic_reference.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do Phase 1R HMC marginal moments agree with the local quadratic Gaussian reference implied by the accepted scalar geometry/mass handoff in HMC `u` coordinates? |
| Baseline/comparator | Local quadratic reference `q(u)=N(m_u, C_u)` derived from the 2026-07-08 geometry/mass artifacts. |
| Reference construction | For local log density `c + l_z^T z - 0.5 z^T K_z z` and `z = F u` where `F = chol(M_z)`, construct `K_u = F.T @ K_z @ F`, `C_u = inv(K_u)`, and `m_u = C_u @ F.T @ l_z`.  The handoff should make `C_u` close to identity, but the formula must not assume exact identity. |
| Primary criterion | Phase 1R artifact passes; reference matrices are finite/SPD; no invalid telemetry; max absolute marginal mean error versus `m_u` is at most `0.5`; all marginal standard-deviation ratios versus reference are in `[0.5, 2.0]`.  These are loose engineering screens calibrated to the local reference scale `std_ref ~= 1` and the current short-run sample size `N=192`; they are promotion vetoes, not statistical proof of agreement. |
| Veto diagnostics | Invalid reference, nonfinite reference, missing Phase 1R seed summaries, nonfinite Phase 1R summaries, Phase 1R vetoes, post-hoc criterion change, positive native divergence when available, treating unavailable native divergence as zero, or unsupported broad posterior claim. |
| Explanatory diagnostics | Marginal mean errors, marginal standard-deviation ratios, reference covariance identity error, Phase 1R acceptance/log-accept tails, and native-divergence availability status. |
| Not concluded | Exact posterior correctness, HMC convergence, HMC readiness, zero divergences if unavailable, sampler superiority, default readiness, GPU/XLA readiness, or source faithfulness. |

## Forbidden Claims And Actions

- Do not use a reference artifact whose target differs from the HMC target.
- Do not call this an exact posterior reference.
- Do not promote descriptive differences without the predeclared agreement
  screen.
- Do not change the agreement criterion after seeing Phase 1R results.
- Do not run GPU/XLA in this phase.
- Do not claim zero divergences unless native divergence telemetry is available
  and all native divergence indicators are false.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Comparator is the local quadratic geometry/mass reference, not HMC readiness or an exact posterior. |
| Proxy metrics promoted | Marginal mean/std agreement only tests local quadratic consistency; it cannot establish posterior correctness. |
| Missing stop conditions | Any invalid reference, Phase 1R artifact veto, agreement failure, telemetry mismatch, or unsupported claim stops before Phase 3. |
| Unfair comparison | No sampler ranking occurs.  HMC is compared only to the geometry it inherited. |
| Hidden assumptions | The local quadratic approximation may be poor; failure triggers localization/geometry repair, not rejection of HMC as a research direction. |
| Stale context | Phase 1R artifact and 2026-07-08 geometry/mass artifacts are explicit inputs. |
| Environment mismatch | This is CPU-hidden artifact analysis and cannot support GPU/XLA/default readiness. |
| Artifact mismatch | JSON/Markdown/result must carry reference formulas, artifact paths, metric roles, manifest fields, and nonclaims. |

Audit status: `PASSED_FOR_CODEX_SUBSTITUTE_REVIEW_BEFORE_RUNTIME`.

## Threshold Rationale

The Phase 2 thresholds are deliberately loose because Phase 1R has only three
short chains and 192 retained samples total:

- Reference marginal standard deviations should be close to one in `u`
  coordinates because Phase 2 inherits the geometry/mass whitening handoff.
- A max marginal mean error of `0.5` is a half-reference-standard-deviation
  engineering screen.  It is not an uncertainty interval and does not establish
  posterior correctness.
- Standard-deviation ratios in `[0.5, 2.0]` catch gross under/over-dispersion
  relative to the local reference while allowing short-chain Monte Carlo and
  local-quadratic approximation error.
- Failure of this screen triggers localization of geometry, transform, or
  short-chain behavior before GPU/XLA.  It does not reject HMC as a research
  direction.

## Exact Next-Phase Handoff Conditions

Advance to Phase 3 only if Phase 2 passes and the Phase 3 GPU/XLA subplan is
refreshed and reviewed.  If Phase 2 fails, write a Phase 2 result and draft a
repair/localization subplan; do not run GPU/XLA as a substitute for reference
agreement.

## Stop Conditions

Stop for invalid reference, agreement failure without planned repair, missing
input artifacts, missing uncertainty/diagnostic role discipline, telemetry
semantics mismatch, unsupported claim, or review nonconvergence.
