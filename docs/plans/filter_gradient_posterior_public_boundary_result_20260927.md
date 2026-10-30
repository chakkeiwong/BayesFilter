# Posterior public derivative boundary audit

The current posterior initializer does not provide a coherent differentiable
public boundary. On the accepted path, calling it under an outer GradientTape
raises LookupError while constructing geometry, because XlaSelfAdjointEig has
no registered gradient. Stopping gradients on supplied start/scale inputs alone
does not prevent that construction error. An invalid-target return instead
exposes identity gradients for the supplied center and scale. These are not
analytical initializer scores or verified total derivatives.

The oldest-original 3582b4ac `_vector`, result constructor and `_build_result`
convert values through NumPy; that boundary disconnects derivatives. The
current tensor conversions introduced accidental exposure to the caller's tape.
The source excerpts and full file hashes are preserved in the audit records.
No NumPy runtime dependency is being restored.

CPU 04384 exposed the original tape error. 04385 preserved that error and showed
the narrower stopped-input hypothesis also fails. CPU 04386 and GPU 04387
capture both failures and complete the ordinary and isolated-boundary reference
comparisons. An accepted stationary Gaussian with default factor_max=2 returns
the same complete payload, including eigen summaries, and exactly 48 physical
target rows under ordinary, repeated and diagnostic tf.init_scope calls. The
invalid case retains its full payload and exactly one target row. The isolated
boundary exposes no start/scale derivative for any returned tensor field.

The reference scope is explanatory. Wrapping the existing Python controller in
it is not the execution repair: the candidate must construct its persistent
owner outside the outer tape and perform numerical work in one enclosing XLA
function, returning frozen tensors. The full endpoint must also compute payload
eigen summaries natively; the current `_eigen_summary` remains eager. No runtime
source was changed by this audit.

| Decision | Primary criterion | Veto / uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Restore the original nondifferentiable initializer boundary during integration | Pinned oldest source disconnects inputs/results; isolated current calls preserve full values and counts | Native enclosing implementation still absent | Isolate owner construction and qualify frozen native outputs | A new analytical gradient |
| Reject stopped-input-only repair | Both CPU/GPU accepted paths still raise during construction | Scope must cover graph-owner construction | Test construction under an outer tape before public dispatch | General autodiff support |
| Include native eigen reporting in integration | Public payload currently executes eager eigensummaries | Larger/ill-conditioned reporting comparisons remain required | Preserve complete payload checks and existing criteria | Whole-initializer or main readiness |

Skeptical review: source-level NumPy conversion alone would not prove current
value parity; the full ordinary-versus-isolated reference calls supply that
check. The isolated diagnostic is not a substitute for native public integration.
No exception was reclassified as a successful differentiable call. The two
failed attempts and harness snapshots remain evidence of rejected hypotheses.
This unit used three CPU and one GPU worker under its 3/2-worker, 900-second
reservation. The receipt is `posterior-public-boundary-verification-04387.json`.
