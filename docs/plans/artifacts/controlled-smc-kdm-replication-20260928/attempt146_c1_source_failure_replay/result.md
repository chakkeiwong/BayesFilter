# Frozen source failure replay

Verdict: SOURCE_PRECISION_AND_COVARIANCE_INDEFINITE. The unchanged source and observation-only copy both
reproduce the same covariance error from the saved pre-case RNG state.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Source failure reproduced | Both arms match expected error | No input or replay mismatch; intrinsic author failure is accepted baseline behavior | Matched port behavior and MATLAB compatibility | Compare the TensorFlow consumer on these exact reference inputs | Author-method repair, paper refutation or numerical repair acceptance |

| Inference status | Finding |
|---|---|
| Hard veto screen | Reference fails its Gaussian covariance domain; this is an accepted reference outcome, not a replication veto |
| Statistically supported ranking | None; deterministic replay |
| Descriptive differences | Matrix diagnostics in summary.json |
| Default readiness | No change or promotion |
| Next evidence needed | Matched TensorFlow comparison on valid reference cases and this intrinsic failure |

The source full-quadratic branch forms Q=Transition.Ginv+2*A and G=inv(Q).
Gaussian proposal precision must be positive definite. Saved matrices are the
actual failing consumer inputs; eigenvalues use the symmetric part and
asymmetry is reported separately. The matrices were never substituted back into
the source. Matrix spectra identify where invalidity is visible; they do not
by themselves establish why the preceding least-squares fit produced it.

The strongest alternative explanation remains Octave/MATLAB solver semantics.
This replay contains no TensorFlow or XLA path, adds no covariance protection,
and supplies no stochastic ranking or full replication result.
