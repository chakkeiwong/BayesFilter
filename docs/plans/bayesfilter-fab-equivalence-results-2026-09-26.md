# FAB-JAX / TensorFlow equivalence result

The pinned author implementation was executed for the first time in an
isolated Python 3.11 CPU environment. The reference is the actual checkout at
`fab-jax-c9f9913`, revision
`c9f991366ca94b2678a7ed620bc9e12655cfef1d`; no independent JAX rewrite was used
for the FAB operations. The plan and skeptical review are in
`bayesfilter-fab-equivalence-plan-2026-09-25.md`.

## Decision

The TensorFlow port is **equivalent to the pinned JAX algorithm over the tested
finite common-draw domain** for FP64 and for FP32 with TF32 disabled. This
includes the canonical nonlinear IAF callback, bridge, AIS increments and
states, both mutation operators, replay, detached gradients, Optax-equivalent
Adam moments/updates, invalid-row boundaries, buffer wrap/adjustment and
complete iteration/checkpoint states. The result is a bounded numerical
equivalence claim, not a universal proof and not evidence of a trained q20 map.

The production TF32 path is **not parity-certified by the declared screen**:
seven four-dimensional FP32/TF32 entries exceeded the predeclared relative and
absolute bound. The largest reported discrepancies were 4.01e-4 in a parameter
gradient and 6.23e-5 in a fresh gradient, against the FP32 screen of 2e-4 for
composed values and 2e-5 for gradients. The compiled complete-iteration checks
still ran; the failures are in the callback/gradient screen.
These are precision-rounding differences observed after algorithmic parity had
passed without TF32; they do not identify a source-operation mismatch. TF32
must remain a numerical qualification requiring a separately reviewed tolerance
or a no-TF32 reference mode for claim-bearing parity.

## Evidence

| Evidence | Result | Artifact |
|---|---|---|
| Pinned upstream test functions | 7/7 passed after visible test-only compatibility repairs; the original suite was also run and exposed its stale import, tuple defaults, strict zero-atol flow assertion and invalid buffer request | `upstream-tests-compat-05.json`, `upstream-tests-01.json` |
| FP64 common-draw complete comparison | 1,804 checks passed, including HMC/Metropolis stress mutations and four replay/fresh iteration lanes | `tensorflow-fp64-stress-03.json` |
| FP64 four-parameter width-16 map | 1,763 checks passed | `tensorflow-fp64-4d16-01.json` |
| FP32 XLA GPU, TF32 disabled | 1,763 checks passed on GPU 1 with memory growth | `tensorflow-gpu-fp32-no-tf32-03.json` |
| FP32 XLA GPU, TF32 enabled | Failed 8/1,763 declared screens; discrepancies recorded above | `tensorflow-gpu-fp32-4d16-02.json` |
| Invalid rows, extreme weights, clipping, buffer boundaries | 21 deterministic checks and 12 bounded RNG-law checks passed | `tensorflow-edges-02.json`, `jax-edges-02.json` |
| Existing local focused regressions | 16 passed | `focused-tests-02.log` |

The common-draw driver exposes all random values at the mutation boundary. It
compares positions, momenta, acceptance/adaptation, every log-weight increment,
stage position, gradient, optimizer moment and parameter, replay index/order,
priority and stored density. It does not pretend that equal seeds imply equal
JAX and TensorFlow random streams.

The native-RNG check used 65,536 draws and the declared family radius. It tests
the six unordered replay-pair events, three normal-CDF events and three
uniform-CDF events against the analytic laws and each other. It establishes a
bounded event-frequency screen only; it does not establish tail or bitwise RNG
equivalence.

## Upstream test limitations and repairs

The untouched upstream invocation is preserved because it is evidence about the
source checkout, not a pass/fail gate for the port. It failed before useful
checks because the test suite imports missing `fabjax.utils.logging`, the flow
configuration has accidental singleton-tuple defaults and its default act-norm
path rejects the test's non-identity initialization. The buffer test requests
24 rows while its configured minimum is 15. The compatibility run made only
test-harness changes: alias `loggers`, unwrap those tuple defaults, disable the
unsupported act-norm branch for that test, use a 1e-5 test-only inverse-density
tolerance for upstream FP32 roundoff, and request 12 rows. The author
sampling, replay, loss or optimizer source was not edited. SMC plots are
reported as visual diagnostics; they do not contain numerical correctness
assertions.

## Repairs made to the TensorFlow authority

- Fresh FAB loss now uses the author's normalized-weight mean. The earlier
  implementation used a sum and was exactly batch-size scaled relative to the
  JAX function.
- `FABAdam` implements Optax Adam's update equations, moment state, bias
  correction and epsilon placement. Checkpoint schema is now v2 so old Keras
  optimizer states cannot be silently resumed.
- AIS accepts supplied common mutation draws, records every stage increment and
  position, and uses the author's valid-row initial replacement operation. An
  all-invalid initial batch still fails closed.
- The replay loss is an explicit shared function, and invalid insertion and
  adjustment behavior is tested against the author boundaries.
- The optimizer counter is placed with the parameters so the XLA GPU update does
  not attempt to access a CPU resource.

## Decision table

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept finite-domain source equivalence | FP64 and FP32/no-TF32 component and state checks pass | No deterministic mismatch remains in tested domain | Unseen target/architecture/precision branches | Keep regression fixtures and rerun after changes | Universal equivalence, q20 fit quality |
| Do not certify TF32 parity | Declared FP32/TF32 screen failed 8 entries | Numerical screen veto only | TF32 tolerance could be scientifically acceptable but was not predeclared | Review precision policy or use no-TF32 for parity claims | TF32 algorithmic wrongness |
| Retain upstream tests as compatibility evidence | 7 functions execute after visible repairs | Original suite is stale/incomplete | Visual tests and patched tolerances are descriptive | Preserve both raw and repaired outcomes | Upstream suite completeness |
| Resume FAB q20 training | Not assessed here | No training run was started by this plan | Deadline and prior campaign state remain separate | Require its own authorized continuation | Trained Neutra, posterior/HMC readiness |

Post-run red team: the strongest alternative explanation is a shared error in
the TensorFlow and diagnostic IAF callback. The parameterized callback was
checked against the same canonical TensorFlow map and the author operations were
called directly, but this does not prove global IAF correctness. A discrepancy
on a new architecture, target status route, or untested invalid pattern would
overturn the broad claim and narrow the equivalence domain; it would not by
itself reject FAB.

## Reproducibility

Artifacts are under
`docs/plans/artifacts/neutra-fab-equivalence-2026-09-25/`. The isolated checkout
is `/tmp/BayesFilter-neutra-fab-20260925` on branch
`codex/neutra-fab-20260925`. The final source hashes before commit are recorded
in the artifact JSON. CPU runs hide GPUs and disable JAX preallocation; the FP32
GPU run used GPU 1, XLA and verified memory growth. No q20 optimizer update,
posterior estimate or external publication was performed.
The complete command, environment, hardware, policy and source-hash manifest is
`equivalence-manifest.json`.
