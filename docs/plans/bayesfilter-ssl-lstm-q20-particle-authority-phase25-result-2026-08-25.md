# Phase 25 Result: q=20 LEDH Density-Adapter Probe

Status: `DIRECT_FULL_STATE_LEDH_BLOCKED_SINGULAR_MEASURE_REDUCED_REPAIR`

The command completed and wrote:

`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase25-attempt1/`

The measured dimensions are state `60`, innovation `20`, deterministic state
`40`, and parameter target `4`. The transition innovation Jacobian induces
`G Q G^T` with rank `20` at all six tested batch/point locations for the
diagnostic tolerances `1e-12`, `1e-10`, and `1e-8`. Its minimum eigenvalue is
zero. A generated transition has zero deterministic residual, while a
one-coordinate deterministic-state perturbation has residual norm `1`.

The explicit Gaussian density in innovation coordinates is finite and the
aggregate UKF target value/score is finite. These are different measures:
there is no ordinary full-state Lebesgue transition density and no checked
identity from the innovation/state measure to the four-parameter posterior.
Consequently direct full-state q=20 LEDH admission is blocked. A reduced
coordinate investigation remains possible; it must carry its own proposal
density and determinant and cannot be called source-faithful parameter LEDH
without a target-binding proof.

## Decision table

| Decision | Primary criterion | Veto diagnostic | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Block direct full-state LEDH | rank `20 < 60` and separate parameter measure | singular-support veto | reduced map not tested | execute Phase 26 reduced-coordinate fixture | no claim about ETPF/SMC/NeuTra or reduced route |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | finite/status/source checks passed; singular full-state measure vetoed |
| Statistically supported ranking | not applicable |
| Descriptive-only differences | eigenvalues, ranks, and residual norms |
| Default-readiness | direct LEDH not eligible |
| Next evidence needed | reduced-coordinate density/Jacobian plus target binding |

## Red team

The strongest alternative explanation is that a carefully defined manifold
measure or reduced innovation flow could still be useful for latent-state
filtering. That would be a different explicitly defined target and does not
repair the current four-parameter particle-authority claim automatically.
Overturning evidence would be an exact common-measure change-of-variables
identity for the declared q=20 target. No such identity was computed here.

## MathDevMCP boundary audit

MathDevMCP classified the report wording as requiring a clearer claim boundary
and the code comparison as scope-limited. It did not certify the runner as a
general proof. The rank statement is supported here by the explicit algebraic
inequality `rank(G Q G^T) <= rank(G) <= 20 < 60`; the artifact is only the
finite numerical instance check.
