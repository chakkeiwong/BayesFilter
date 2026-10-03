# NeuTra implementation consolidation: result and recovery note

The reviewed implementation plan has been executed. The active numerical
authority is `bayesfilter/inference/neutra_transport_core.py`; the configured
public API is `bayesfilter/inference/neutra_transport.py`. The implementation
reference is `docs/reference/neutra-implementation.md` and the plan/review is
`docs/plans/bayesfilter-neutra-single-authority-plan-2026-09-24.md`.

## Changes and source correspondence

Ordinary, weighted, legacy and frozen IAF classes now delegate masks, networks,
scale conventions, inverses and score derivatives to the same core. Their
constructors retain old variable ordering, seeds, named extensions and saved
forward-map conventions. Composition, affine wrappers and scalar canaries also
delegate their numerical primitives. Scale-aware kernels and the varying-Hessian
affine wrapper use this authority. Repository discovery found no additional
IAF numerical classes in active benchmarks, scripts or experiments. Historical
execution snapshots and author reference code were preserved.

The source IAF profile implements the paper architecture, author block masks,
author variance-scaling initialization and the author-code option
`b+c*tanh(h/c)`. The new NAF is a full conditional DSF. Its source profile
retains cMADE's context scale/bias parameters, final conditioner direction
normalization and shared output projection. A constant context specializes
that source to the unconditional posterior. Full-support log-domain sigmoid
mixtures implement the paper equation without the support-changing shrinkage
in the author numerical code. Source anchors and deviations are explicit in
the implementation reference.

Vaitl's general layerwise score recursion is connected to batched Adam via an
actual path-gradient option. It reuses each layer's forward evaluation and
solves triangular systems for IAF/NAF. The coupling-specific O(d) algorithm is
not claimed. The inverse uses bracketed coordinate solves and implicit input
and parameter differentiation. Standard RKL remains available as a configured
estimator. Actual RKL loss is separate from the gradient carrier. Target
callbacks supply values and first scores; no target Hessian is requested.

The new format saves the full map configuration, including conditioner
parameters, optimizer state, target identity and ordinary hashes. The existing
frozen loader, HMC geometry builder and source closure understand it. Weighted
training accepts a configured map, and tempered checkpoint reconstruction
supports the map and its reference-affine wrapper. Training finalization runs
the standard 1,000-point diagnostic; NAF diagonal log derivatives are not
mislabeled as bounded-scale saturation.

The audit repaired a real derivative defect: the historical frozen
`dsge_bounded_tanh` map is `c*tanh(h)` but its score implementation omitted the
factor c. Its forward map remains the same. Prior score-based conclusions for
that variant need rechecking. This is not evidence that every earlier q20 map
used that variant; the q20 weighted map used normalized bounded tanh.

## Verification

Artifacts are under `docs/plans/artifacts/neutra-single-authority-2026-09-24/`.
`run-manifest.json` records commands, source hashes, environment and outcomes.

* Main CPU/XLA analytic and compatibility suite: **198 passed**, 277.64 seconds.
  Includes 27 new authority tests, legacy training/serialization, affine and
  dense frozen readers, weighted and tempered routes, mechanisms, scale-aware
  training, graph contracts and post-training diagnostics.
* Additional compatibility suite: **35 passed**, 15.47 seconds; three tests
  requiring an unavailable external historical checkout were excluded explicitly.
* Final HMC documentation and changed scale-aware wiring check: **16 passed**,
  15.05 seconds. This includes one repeated scale-aware regression, not 16 new
  unique tests beyond the main suite.
* Final scale-aware source-identity/checkpoint checks: **6 passed**, 23.77
  seconds, after adding the shared numerical authority to its source identity.
  These repeat relevant main-suite checks on that final source change.
* Final GPU/XLA smoke, `gpu-02.json`: **passed**, 28.19 seconds on GPU 0,
  float64, TF32 disabled, verified memory growth. Both source profiles accepted
  four batched path-gradient updates, with one evaluate trace and one train
  trace each. Maximum absolute CPU/GPU gradient differences were
  `1.29e-16` (IAF) and `1.02e-16` (NAF). Inverse round-trip errors were
  `4.44e-16` and `2.17e-11`. This tiny fixture is not a scaling benchmark.
* GPU allocator peaks were 97,792 and 212,480 bytes, respectively, within this
  tiny process; these are allocator live peaks, not device reservation claims.
  The earlier engineering revision used 22.69 seconds (`gpu-01.json`), for
  **50.88 seconds total GPU-smoke worker wall time**, within the 300-second
  implementation allowance. The prior q20 diagnostic allowance was not reused.
* The LaTeX repair discussion builds as a seven-page standalone chapter.
  The rendered new section on page 6 was inspected for layout and prose. The
  main mathematical discussion was preserved; the new section explains what
  changed and what remains to validate.

New numerical checks include nonidentity triangular Jacobians and logdet,
author mask/conditioner equations, free-bias gradients outside the cap, NAF
full-support tails, inverse input and parameter derivatives, exact-Gaussian
zero path gradient, one-forward-per-layer traversal, Adam continuation,
invalid-input rejection, full rollback of overflowing proposals, immutable
frozen restore, actual HMC geometry construction, weighted inverse-density
training and tempered reconstruction. CPU fixtures deliberately hide GPUs.

Three external-reference tests could not run: they pin dsge_hmc commit
`d94566c9f70b3143e599a56eba7cb461ff2bda88`, whereas the sibling checkout is
`aa6fa3d13b5b68bc108b49f94660e040539bf927`. An isolated local clone could not
read the pinned tree, and a fresh isolated remote fetch returned `not our ref`.
The sibling working tree was preserved. These checks are unavailable, not
passing. Local frozen/training parity and independent derivatives passed.

The first durable HMC replay invocation passed seven of eight cases, including
both new transport families with both graph modes. The older affine fixture
had no verified survivor with its narrow engineering acceptance band; receipts
reported no hard numerical vetoes. The test's purpose is geometry and durable
replay, so its band was made explicitly broad, centered at 0.7, leaving all
runtime/default scientific policies and hard vetoes unchanged. The final replay
outcome is **8 passed**, 215.70 seconds, in `pytest-hmc-consumer-02.xml`.
All four supported families (old affine, old dense IAF, configured source IAF,
configured conditional NAF) passed both graph modes through public tuning,
retained-runner export/reload and short retained replay. This is explicitly a
mechanics fixture with a broad acceptance band, not posterior qualification.

## Terminal review and decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Use the shared implementation for configured NeuTra maps | Caller wiring, independent mathematics and serialization checks pass | No unresolved local map/gradient failure in the reported suites | Three pinned external-reference checks unavailable | Preserve compatibility tests and core-only numerical changes | Bitwise reproduction of historical paper experiments |
| Permit source IAF and conditional NAF as implementation choices | Source-profile, batched gradient, inverse and frozen-consumer checks pass | Nonfinite/invalid updates and failed inverses reject | q20 training protocol and learned geometry remain uncalibrated | Plan target-specific calibration before training claims | A correctly trained q20 NeuTra map |
| Permit path estimator as a configured training option | Layerwise recursion and actual Adam update verified | Missing/nonfinite gradients reject; rollback verified | Target-specific variance and runtime scaling | Measure under a declared target protocol | Universal variance reduction or coupling O(d) complexity |

| Inference status | Finding |
|---|---|
| Hard veto screen | Local engineering invariants pass; unavailable external reference is disclosed separately |
| Statistically supported ranking | None; this work did not compare trained stochastic methods |
| Descriptive differences | Tiny engineering runtimes and CPU/GPU errors only |
| Default readiness | One code authority established; no new q20 scientific hyperparameter default promoted |
| Next evidence needed | Target-specific training calibration, fresh 1,000-point diagnostics, and downstream HMC qualification |

Terminal source review checked the actual paths from training, frozen loading,
weighted density and HMC consumers. It also checked backward compatibility,
parameter hashing, source closure, failed-inverse behavior, implicit derivatives,
no pfor or sample-wise target loops, and separation of source mechanisms from
local choices. Review was performed locally; no independent reviewer was used.
The strongest remaining alternative explanation for poor q20 whitening is
still inadequate target-specific optimization/capacity, not an established
failure of NeuTra. Successful engineering checks do not overturn that concern.

Recovery: no q20 research campaign was launched or resumed; no commit or push
was requested for this task. Existing dirty campaign files and archived runs
were preserved. Continue with a separately scoped q20 training/calibration plan
using the configured API, then evaluate the same frozen map downstream.
