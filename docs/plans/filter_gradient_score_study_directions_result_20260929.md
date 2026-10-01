# Gaussian analytical direction consumer repair

The five Gaussian score-study proposals now assemble their complete analytical
score in one fixed-signature XLA owner. `evaluate_gaussian` no longer dispatches
six parameter directions from Python. LEDH (with and without diagnostics),
SGQF covariance, mixture covariance, integrated KDM and resampling KDM all call
the same TensorFlow direction loop through their existing configuration-specific
factories. The mathematical filtering and analytical tangent authorities are
unchanged. No autodiff score, scalar training fallback or canonical admission is
introduced. The direction axis is a parameter axis, not a training-sample axis.

The owner returns every nested diagnostic result plus native finite/valid and
value-invariance flags. The endpoint enforces those flags on the host because
XLA can discard assertions. Optional diagnostic failures retain their original
payload before generic validity rejection. Model tangent contractions are
explicit tensor operations; covariance-diagnostic eigenvalue reductions are
batched across the three covariance families. The mixture factory declares
the fixed horizon and particle dimensions of existing auxiliary tensors.
Existing bounded immutable-configuration caches retain owners; no new arbitrary
callable cache or compiler-memory eviction promise is made.

Runs04756–04777 use the frozen, fresh September fixture in
`tests/fixtures/filter_repair_score_directions_20260929.json`, SHA256
f3d079e681f2f5202cdc487e16838a8f71e44e6db1c3b64b1fa5bf6dd1106f3c.
The scope is d2,o1,N8,T2,float64, exact K=N, with all arrays and controls frozen
before execution. References invoke the same directional authority through
test-only Python loops. Resampling KDM derivatives replay fixed samples,
proposal densities and component indices to differentiate its own finite scalar.

| Consumer | Final CPU run | GPU run | Maximum CPU/GPU five-point error |
|---|---:|---:|---:|
| LEDH | 04766 | 04771 | 7.29e-12 |
| LEDH with diagnostics | 04767 | 04772 | 7.29e-12 |
| SGQF covariance | 04768 | 04773 | 5.86e-12 |
| Mixture covariance | 04769 | 04774 | 7.61e-12 |
| Integrated KDM | 04764 | 04775 | 7.87e-12 |
| Resampling KDM | 04765 | 04776 | 9.49e-12 |

All complete nested output comparisons are exact on each device, including
changed theta, data and parameter directions. Five-point steps2e-4 and1e-4
pass the unchanged absolute2e-6 gate. Every case verifies one trace, enclosing
HLO without pfor/host callbacks, actual endpoint wiring, cached-owner reuse,
nonfinite-input rejection and the correct host exception. Generic checks also
exercise nested integer/boolean diagnostics, per-direction invalidity,
value-invariance failure and an explicit non-JIT CPU reference. GPU3 is a
non-display device with trusted provenance and verified growth before logical
initialization; TF32 is enabled. Another campaign's retained context excludes
uncontended GPU timing or capacity claims.

04777 passes161 source-bound readback/policy checks. The guard now covers307
sources with1442 exact exceptions: six new exceptions read configuration or
pack existing tensor fields; none permits numerical Python iteration. This
is still a scoped guard, not proof of whole-repository compliance.

Preserved failures and repairs:

- 04757: the test wrongly assumed zero residual design must be rejected. The
  actual reset checks gap/target/injected Cholesky factors, which may remain
  valid with transported covariance and ridge. Zero-design status/value parity
  is retained, and a nonfinite design tests rejection. No numerical gate changes.
- 04761/04762: fixed auxiliary dimensions were lost in TensorArray/trace shape
  inference. Construction failed closed; explicit horizon and particle shape
  bindings repaired it without dropping diagnostics or using dummy evaluation.
- Four previously passed CPU cases were refreshed after preserving the original
  diagnostic-error payload ordering. The final endpoint check includes actual
  host rejection rather than relying only on returned status tensors.

The22-worker numerical/readback allocation consumed452.237632 CPU and229.995186
GPU process-seconds, within its3600/3600-second limits. Exact commands, source
hashes, TensorFlow2.19.1 environment, seeds, devices, time and outputs are in the
numbered manifests. The successful first readback unnecessarily loaded all old
manifests; its lookup is now bounded to this unit and will be rechecked with
the cost readback. That reporting overhead is not a filtering memory result.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Accept this bounded execution repair | Complete parity, analytical derivatives and actual CPU/GPU consumers pass | No remaining numerical/execution failure for this fixture | Scale, cost, native compiler residency | Fresh-process owner cost screen | Whole-repository or canonical LEDH completion |
| Keep performance open | No matched cost result yet | Cost acceptance pending | Setup, warm costs and diagnostic-output residency | Execute `filter_gradient_score_direction_cost_20260929.md` | Speed or capacity improvement from correctness alone |

Skeptical terminal review: finite-program derivative agreement and compilation
do not certify canonical LEDH, approximation quality, HMC, training eligibility,
large-particle feasibility or posterior correctness. A different model/shape or
control regime could invalidate the observed compatibility. The shared assembly
reduces numerical forks, but its loop still repeats the underlying filtering
work for six directions. Nonlinear EKF/UKF and LEDH-family consumers still have
their own host direction loops and are explicitly queued in
`filter_gradient_nonlinear_directions_20260929.md`. This result does not close
F07/F08/F19 as whole findings or authorize merging main.
