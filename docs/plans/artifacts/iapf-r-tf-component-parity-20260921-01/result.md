# R–TensorFlow component comparison

All eight numerical comparisons pass. The largest absolute disagreement is
6.22e-15, below the predeclared 1e-10 absolute/relative tolerance. Four
consumer-wiring and endpoint checks also pass. These results support the
shared Gaussian calculations in the tested FP64 fixture; they do not establish
full-filter equivalence.

The compared quantities are the Gaussian-plus-floor normalizing integral,
mixture probability, both proposal branches under supplied noise, profiled
density loss and its analytical gradient, the relative shape residual, and
the relative-shape objective and gradient. The density normalizations and
log-variance/log-standard-deviation coordinates were matched algebraically
before comparing. The R numerical reference and actual TF functions execute
from captured source files. No NumPy, autodiff or pfor was used.

The consumer checks verify the factory imports and their shared-function
bindings. An actual call to `execute_iapf` with d=o=2 raises its existing
scalar-only restriction. This is the expected capability result, not a
successful multidimensional filter call. The unrestricted/full-filter
call-chain question is therefore answered negatively for this endpoint.

| Component or scope | Result | Interpretation |
|---|---|---|
| Shared Gaussian twisting formulas | Pass, maximum error 6.22e-15 | No arithmetic disagreement found on the correlated fixture |
| Both profiled fitting objectives and analytical gradients | Pass, maximum error 3.47e-16 | Same defined quantities agree after coordinate/scale conversion |
| Consumer-to-shared-factory wiring | Pass | The inspected consumer resolves these implementations |
| Multidimensional full-filter endpoint | Explicitly rejected by the real endpoint | The shared primitives do not imply multidimensional consumer support |
| Whole R/TF filter equivalence | Not checked; current scopes differ | Model, fitting family, floor and controller must be matched first |

The current TF model has A=theta[0]I plus a single upper-shift band, with
separately structured Q, H and initial covariance. It differs from the first
paper study's dense A[i,j]=0.42^(abs(i-j)+1), identity noises and initial
covariance. The density-fit consumer is scalar; the existing separate
log-quadratic comparator fits full precision matrices, whereas the qualified
R alternative uses diagonal precision. These differences are explicit model
and method choices, not evidence that the shared Gaussian algebra is wrong.
No production or candidate code was changed to conceal them.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Limit |
|---|---|---|---|---|---|
| Retain shared primitives as comparison tools | All checked values/gradients agree | No numerical or wiring veto | Only one deterministic fixture and CPU FP64 tested | Use these checks during a specifically matched consumer adaptation | No whole-filter or GPU claim |
| Do not label current full filters equivalent | Model/fitter/endpoint identities differ | Same-method comparison prerequisite is unmet | Original-paper numerical fitting choices remain unknown | Resolve the replication target and explicitly match the consuming model/method | No silent objective substitution |

| Inference status | Finding |
|---|---|
| Hard veto screen | No common-primitive defect; actual scalar restriction confirmed |
| Statistically supported ranking | N/A: deterministic comparison |
| Descriptive-only differences | Runtime and model/implementation scope differences |
| Default readiness | None; CPU non-JIT reference exception only |
| Next evidence needed | Matched model, fit family, floor, controller and complete consumer execution |

Red-team note: a shared defect or an untested branch could survive parity.
The independent R derivations and conformance tests reduce that risk, but
these eight comparisons cannot certify the full adaptive particle filter,
parameter score, GPU/XLA route, canonical LEDH, KDM or HMC.

One attempt completed in 3.837859605 worker seconds, no repair or retry.
Environment: TensorFlow 2.20.0-dev0+selfbuilt, FP64, intentionally CPU-only,
stable-signature tf.function with JIT explicitly disabled for reference.
Command, captured sources, commit, environment, deterministic fixture hashes
and outputs are preserved in `attempt01/manifest.json`; numerical results and
wiring verdicts are in `attempt01/parity.json`. No process remains running.

Of 180 transferred seconds, 176.162140395 remain. Together with the
691.860608825 unspent reserve, the same authorized pool has 868.022749220
seconds left. Exact original-paper replication remains scientifically
blocked by the unresolved fitting specification, not by a phase-approval
requirement or an exhausted allowance.
