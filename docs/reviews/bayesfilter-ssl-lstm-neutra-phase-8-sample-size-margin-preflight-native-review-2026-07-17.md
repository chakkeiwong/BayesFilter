# Phase 8 Sample-Size And Margin Preflight Native Review

Date: 2026-07-17

Verdict: `AGREE_GPU_SMOKE_ONLY`

## Scope

- `docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-sample-size-margin-preflight-plan-2026-07-17.md`
- `docs/benchmarks/run_ssl_lstm_neutra_phase8_sample_size_margin_preflight_2026_07_17.py`
- `tests/test_ssl_lstm_neutra_phase8_sample_size_margin_preflight.py`

## Findings And Repairs

1. The first draft's smoke loop generated only the decisive persistent
   variance-ratio `1.05` family but attempted to evaluate all families. The
   loop now iterates only over the explicitly generated pilot-family mapping,
   with a regression test.
2. The installed TFP `StudentT` constructor requires explicit `loc` and
   `scale`. The MMD operating-curve call now supplies `float64` tensors for all
   parameters, with a focused execution regression.
3. Historical interval-width recovery originally mixed `float32` quantiles
   with `float64` receipts. All analytical and simulated critical values now
   use `float64`.
4. The MMD effective degrees of freedom is reconstructed directly from the
   same pair/block variance terms used by the production interval, rather
   than numerically inverted from a rounded critical value.

## Audit

| Question | Review result |
| --- | --- |
| Correct failed baselines? | Yes; exact 448- and 1984-draw receipt hashes and failed decisions are mandatory |
| Confirmation leakage? | No retained private path, sample tensor, or confirmation forecast input exists; pilot receipt contributes bandwidth lineage only |
| Complete feature decision? | Yes; joint 20-dimensional normal draws feed all-coordinate TOST/Bonferroni logic |
| Feature/MMD dependence assumed? | No; combined pass/intersection and material/union probabilities use sharp Frechet bounds |
| Margin selected statistically? | No; all three margins are labeled historical or arithmetic sensitivities, and selection fields remain null |
| Proxy promoted? | No; smoke cannot freeze feasibility, and material feasibility still requires direct finite-sample validation |
| Resource boundary? | Yes; GPU 1, XLA, `float64`, wall cap, fresh seeds, immutable output, no HMC action |
| Main approximation risk? | Joint feature normality plus `1/N` covariance and linearly scaled MMD block degrees of freedom; recorded explicitly and subject to direct validation |

5. The first material-mode draft did not itself bind the passing smoke. A
   nonnumerical material-only gate now requires smoke SHA `7eaf2b17...`, its
   exact pass decision, and null selection fields. The already-smoked numerical
   surfaces are unchanged.
6. The first cost table projected HMC only and ignored the fixed 64-draw pilot,
   256-draw acquisition segments, and forecast cost. It now rounds acquisition
   upward by segment, records unused surplus, and adds the observed target-pilot
   warm forecast rate plus one compile overhead. These remain conservative
   planning estimates, not acquisition authority.

Focused checks after this repair: `12 passed`; Python compilation passed; scoped
`git diff --check` passed.

The smoke is authorized only to validate the exact GPU/XLA route, primary
stress-family computation, operating-curve plumbing, trace gates, and strict
receipt. It cannot establish a feasible draw count or margin.
