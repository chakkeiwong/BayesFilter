# Supplied mass implementation and validation

Implemented the reviewed plan in the standalone Walnutpie TF/XLA package.
Existing callers and numerical diagonal-mass adaptation are preserved.

## Behavior

- make_supplied_mass_functions accepts diagonal momentum covariance M and
  epsilon as runtime operands in initialize/fit. Config.adapt_mass=False
  preserves M through warmup; True seeds the existing position/score-moment
  adaptation from M. All sampling calls freeze tuning.
- make_dense_mass_functions supports full SPD supplied M with an explicit
  checkpointed Cholesky basis. Fixed mode preserves physical M. Adaptive mode
  adapts a diagonal mass within that fixed basis, yielding L diag(m) L'; it
  does not learn an unrestricted dense covariance.
- Model data remain runtime operands, model/gradient evaluation stays in XLA,
  and returned dense-interface draws/log density use physical coordinates.
- validate_mass gives readable host-boundary errors. Compiled initialization
  separately sets invalid flags; callers must inspect validity before use.

## Correctness audit

The transform y=L' q, p=L r preserves p' dq=r' dy and converts .5 p' M^-1 p
to .5 r' r for M=L L'. The transformed log density subtracts log det L.
Paper Definition 10 must still use physical coordinates: in transformed
coordinates the stopping dot product is r' diag(inv_y) L^-1 L'^-1 delta_y.
Naively reusing the y-coordinate U-turn would change the specified algorithm.
That error was caught in review and corrected before validation. The private
kernel factory overrides only this stopping calculation; the shared tree,
integrator and adaptation remain the numerical authority.

Dense symmetry tolerance 1e-12 times max(1,element scale) is an FP64 engineering
roundoff tolerance, not a scientific tuning setting. Parity checks use 2e-11
or tighter inherited FP64 engineering tolerances; no statistical inference is
drawn from them.

## Executed evidence

- 34 CPU tests passed in 175.00 seconds: supplied metric lifecycle, source
  reference, continuation, validity, existing initialization/control and R2P.
- Two added dense leapfrog/runtime-data checks passed in 10.97 seconds.
- Two final nonidentity diagonal-vs-dense physical trajectory checks passed
  in 28.37 seconds, covering deterministic and R2P kernels. Combined with
  the first two batches, 38 CPU regression checks passed.
- Eight GPU cases passed in 182.54 seconds: deterministic/R2P x diagonal/dense
  x fixed/adaptive. Every complete fit graph has one trace and XLA compilation.
- GPU allocation growth was verified before initialization on GPU 2, RTX 4080
  SUPER; FP64, TF32 off. Diagonal HLO has no custom calls. Dense HLO uses native
  cuBLAS triangular solves and cuSOLVER Cholesky; these are device linear-algebra
  calls, not CPU sampler callbacks. No claim that the entire graph is one fusion.

Artifacts: docs/benchmarks/artifacts/walnutpie-supplied-mass-20261029-gpu01/
contains result.json, validation-manifest.json, HLO files and source copies.
Focused tests: tests/test_walnutpie_supplied_mass.py. Four separate replication
harness checks passed in 1.25 seconds; they test original-source continuation,
output projection and binary storage rather than mass adaptation.

## Decision table

| Decision | Primary criterion | Veto | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Complete requested optional API | Algebra/lifecycle/CPU/GPU invariants pass | No failed invariant or callback fallback | Larger target performance and posterior behavior unmeasured | Use explicit supplied mass and chosen adaptation policy | Faster sampling or improved funnel convergence |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Focused engineering checks pass |
| Statistically supported ranking | None; this is not a method-ranking experiment |
| Descriptive-only differences | Compile time and individual validation runtime |
| Default readiness | No public/default change |
| Next evidence needed | Target-specific tuning/posterior and steady-state performance study before scientific use |

## Result review

The strongest remaining limitation is the fixed-basis interpretation of dense
adaptation: unrestricted dense adaptation was neither implemented nor tested.
Returned checkpoint coordinates are transformed and must not be interpreted as
physical q without the supplied helper. A failed nonidentity parity or cache
test would overturn the engineering verdict; short Gaussian checks do not
establish general posterior validity. The million-iteration author comparison
uses frozen identity mass and remains scientifically separate.
