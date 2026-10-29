# R4d program and terminal result review

28 September 2026. **PASS for the bounded repair/diagnostic execution;
strict paper-horizon numerical parity FAIL; both full replications OPEN.**
This is source-based self-review, not an independent reviewer certification.

Reviewed the actual R vmmin source (optim.c lines 113–280), the shared TF
optimizer/fitter/backward/controller, exact fixture exporter, semantic and
consumer harnesses, master dispatcher, paper Proposition 1 and Section 5.1,
and saved source/result snapshots. Attempts 35/36 are executable consumer-chain
checks; a standalone optimizer function alone would not have established
wiring.

| Skeptical audit issue | Finding / resolution |
| --- | --- |
| Wrong comparator | Actual R optim with explicit finite controls; no new substitute optimizer or toy model |
| Hidden assumptions | Reconstructed finite 100-step BFGS, full-rank initialization and FP64 remain labeled; original settings unrecovered |
| Proxy promoted to science | Solver/parameter gates measure a finite port, not statistical replication; one close likelihood remains descriptive |
| Missing stop conditions | Phase time ceilings, parent budgets, invalid accepted computations and corrupt/missing evidence stop execution; a failed candidate triggers a defined next question |
| Unfair backend comparison | Identical full-precision data; exact-initializer replay; no backend ranking or timing superiority claim |
| XLA/environment confusion | CPU explicitly hides GPU; GPU graph/XLA use trusted access and verified memory growth; tested XLA signatures trace once |
| Missed call chain | Initial backward-state structure regression caught and repaired; 129 checks pass on actual CPU and GPU/XLA consumers |
| Observability | No-change counter bug repaired and directly tested; full traces, curvature, Hessian symmetry and failed endpoints preserved |
| Numerics-altering policy | No objective, tolerance, damping or maxit retuning; lower-triangle mirroring restores source storage semantics and is regression checked |
| Historical/current mismatch | R4c full adaptive run remains tied to its old snapshot; repaired paper gate fails before adaptive execution |
| Budget realism | Full repeated MDPF study still exceeds remaining resources at observed source training cost |

The retained failed coefficient criterion has not been loosened. The next
downstream question needs its own predeclared statistical error budget and
fresh evaluation data. That is a change in the measured scientific quantity,
not permission to retrospectively rename a failure. The paper's regularized
article recipe and the finite unpenalized reconstruction are distinguished.

All 15 dispatcher regression tests pass; the saved unittest wall time is
2.810 seconds. [Regression result](regression-r4d.json),
[closing validation](final-validation-r4d.json) and
[current source index](closing-source-index-r4d.json) are saved alongside this note. The new status test prevents a repaired-source pass from erasing
historical failed evidence or completing either scientific replication.
No independent agent, external message, package mutation or new numerical
default was involved. Intermittent sandbox mount failures were rerun through
the permitted escalated command mechanism; they are unrelated to XLA.
