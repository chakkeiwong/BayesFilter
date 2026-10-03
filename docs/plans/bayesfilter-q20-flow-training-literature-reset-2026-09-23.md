# q20 flow-training literature: continuation checkpoint

Active request completed: survey literature on failure to learn nonlinear
deformation and parameterization. Read the
[result](bayesfilter-q20-flow-training-literature-results-2026-09-23.md) and
[source ledger](artifacts/q20-flow-training-literature-2026-09-23/source-ledger.json).
Twelve primary papers and available author code are archived under
`.localresources/q20-flow-training-literature-20260923/`.

Recommended mechanisms, not an executed repair: free global scale outside the
conditional cap; a direct smooth monotone scalar deformation first in latent
coordinate 2; exact RKL path gradients to test noise reduction; fixed
conditioner normalization/initialization with measured nonlinear response.
Keep standard and path-gradient estimators conceptually distinct from a new
score-matching objective. Vaitl 2024 Proposition 3.2 computes proposal scores
forward by transposed-Jacobian solves; no target Hessian or training-time
numerical inverse is required. Runtime/variance benefit for our IAF is untested.

Important corrections: zero final initialization is established practice, not
itself a bug. Ideal NAF sigmoid mixtures are smooth/full-range, but the inspected
author code clips the sigmoid mixture and therefore bounds each scalar output.
Use exact log-domain identities if implementing full-range NeuTra. Identical
components at exact identity initially expose only affine scalar directions;
near-identity initialization must retain meaningful shape directions. Standard
rational-quadratic splines are C1, generally not C2, so transformed-score jumps
at knots need checking before HMC use.

NAF's Jacobian determinant is efficient because its autoregressive Jacobian is
triangular and its conditioner can emit all coordinate parameters in one
batched pass. Its inverse is generally numerical, however. It is therefore a
candidate for a reverse-KL mechanism test, not yet a production replacement;
the full result explains the forward/logdet versus inverse distinction and the
current HMC codec gap.

The current HMC public interface only reconstructs supported frozen affine or
dense-IAF payloads. A new scalar family needs serialization, reconstruction,
inverse/Jacobian/score checks and explicit public-tuner support before HMC;
do not pass it as an arbitrary force. See the result's integration note.

No candidate ranking, trained-map certification, or posterior promotion.
No target evaluations, framework imports, GPU work, or scientific experiments
were performed. Campaign and diagnostic balances remain inherited from the
math-audit checkpoint: 143975.28597232018 and 7.711412891243526 seconds,
respectively. The previous request for two extra diagnostic minutes remains
unanswered; literature work did not treat it as approval.

Next action if repair is requested: price and review an endpoint/gradient
mechanism check and a frozen-map scalar-deformation experiment using the
report's derivations. Keep the existing target, identity latent mass, and
standard 1,000-point post-training diagnostic. Inspect the frozen repair tree
for claims about previous execution. Preserve unrelated dirty work. No runtime
code or master program was changed by this survey.
