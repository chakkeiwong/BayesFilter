# Cubature/GenUT Exact-SV N=1000 Feasibility Blocker

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_ENGINEERING_ONLY_CLAIM_REVOKED`

> **Correction, 2026-07-22:** The later “successful claim” used non-DGP
> observations and is revoked.  This handoff is retained only for historical
> execution and memory provenance.

> Historical note: this blocker described two permission-bound attempts before
> a process launched. It was superseded by `feasibility_attempt03` and the
> successful tuned claim in `tuned_claim_attempt02`. Do not treat it as current
> execution state.

## Outcome

The `N=1000`, `T=50` single-seed feasibility harness and reviewed plan are
ready, but the trusted/escalated GPU command was not launched. The managed
permission review timed out twice before process creation.

Checks after both timeouts found:

- no `run_cubature_genut_exact_sv_n_scaling.py` process;
- no N=1000 artifact directory or result file; and
- no scientific or engineering result to interpret.

At the time, the exact pending command was:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_cubature_genut_exact_sv_n_scaling.py \
  --output-root docs/benchmarks/artifacts/cubature_genut_exact_sv_n1000_20260721/feasibility_attempt02 \
  --n-values 1000 --t-values 50 --seeds 3400
```

At the time, the next action was this one feasibility seed only. Fresh N=1000
tuning and the 16-seed untouched claim were conditional on finite GPU output,
verified memory growth and placement, residuals below `1e-2`, and acceptable
allocator peak/wall time.

## Resolution

The feasibility run subsequently completed on the GPU at `N=1000,T=50`. The
fresh-scope tuning and all 16 untouched claim seeds also completed. See:

- `docs/benchmarks/artifacts/cubature_genut_exact_sv_n1000_20260721/feasibility_attempt03/result.json`
- `docs/benchmarks/artifacts/cubature_genut_exact_sv_n1000_20260721/tuned_claim_attempt02/result.json`
- `docs/plans/bayesfilter-cubature-genut-exact-sv-n1000-result-2026-07-21.md`

## Nonclaims

The permission timeout is not evidence for or against N=1000 feasibility,
accuracy, memory use, bias reduction, or score precision.
