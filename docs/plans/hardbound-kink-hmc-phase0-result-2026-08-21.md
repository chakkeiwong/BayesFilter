# Hard-Bound Kink HMC: Phase 0 Result (2026-08-21)

Program: `hardbound-kink-hmc-master-program-2026-08-21.md`.

Built: `bayesfilter/hardbound/{__init__,dns_curve_tf,model_tf,reference_numpy}.py`,
`tests/hardbound/test_phase0_evaluators.py`.

Gates (env `tf-gpu` python, CPU via `CUDA_VISIBLE_DEVICES=-1`, seed 20260821):

- G0.1 PASS: TF vs independent NumPy reference, 100 random states, both
  targets; means to 1e-10 abs, log densities to 1e-8 abs.
- G0.2 PASS after one documented repair: the strict-positivity assertion on
  the softplus-hard gap fails in float64 where the exact gap
  (~alpha*exp(-|u-ell|/alpha)) is below machine resolution (observed
  -6.9e-18 on a far-from-bound row). Repair: tolerance 100*eps for the
  lower edge, upper bound alpha*log2 kept exact, strict near-bound
  attainment (>0.99*alpha*log2 on a pinned state) kept. This is a
  roundoff safeguard in the survey Sec. 4.3.2 sense, not a weakening of
  eq. (88)-(89), which hold in exact arithmetic.
- G0.3 PASS: 1000 random factor states, 400-point s-grid: every binding
  set has <= 2 switches (interval pattern), per the crossing lemma.
- Simulation reproducibility PASS (same seed, identical observations;
  shape [40, 13]).

Suite: 4 passed, ~4 s. No plan amendments needed for Phase 1.
