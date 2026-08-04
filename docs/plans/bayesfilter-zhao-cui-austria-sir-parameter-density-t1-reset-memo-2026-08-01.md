# Zhao-Cui Austria SIR Parameter-Density T1 Reset Memo

Date: 2026-08-01

## State

The centered external-parameter child branch is closed.  The immutable
fixed-variant parent remains admitted for value only:

- parent identity: `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`;
- T1 value: `-31.1290512231882`; and
- T1:T2 cumulative value: `-66.28380350560136`.

No analytical total-score child is admitted.  HMC, T2/T5/T10/T20 score
recursion, and GenUT/SGQF/UKF score comparisons remain closed.

## Evidence

The authority diagnostics passed.  Two independent prefix seeds on 32 fixed
points had paired `z^2` mean `0.601`, maximum paired `z=1.326`, and all
`g_AB <= 1`.  The 8,192-versus-65,536 sample-growth audit on eight points had
maximum paired `z=0.236`, mean `z^2=0.0131`, and ESS minima `8188` and `65504`.
These results rule out the tested gross MCSE failure and obvious proposal
instability as the explanation for the broad-prefix failures.

The terminal rank-12 child was an exact zero-padded expansion of the rank-8
child.  It preserved the origin value and passed the origin point-score gate,
but failed the untouched likelihood score, off-origin mass, normalized shape,
and retained-prefix gates.  Its third likelihood-score standardized residual
was `1.9976`; its retained-prefix residuals were
`[1.1432, 2.6618, 3.7249]`.  The child identity is
`553442f49ddd59b99bafbf0b3e7fde39d6791aa24835413b6e59e36ae93f8368`.

## Interpretation

This is a valid representation/target-coordinate failure, not an authority
failure and not evidence that the admitted fixed values are wrong.  The
current finite child computes

\[
h_\theta(r)=h_0(r)+\sum_k\psi_k(\theta)H_k(r),\qquad
\rho_\theta(r)=h_\theta(r)^2+\tau,
\]

in one static 36-dimensional affine frame.  Its origin is exact, but the
off-origin likelihood and prefix derivatives are not identified by that
origin slice.  Increasing rank within the tested family did not repair this
failure.  The existing UKF-guided proposal is geometry only and is not an
exact conditional authority.

## Boundary

The following are not authorized by this reset:

- rank 16 or another optimizer/temperature for the closed child family;
- off-origin density repair using any failed child as promotion evidence;
- a T1 score claim based only on the parent fixed slice;
- T2/T20 recursion, comparator ranking, or HMC; and
- calling the current child or proposal source-faithful Zhao-Cui.

The next authorized work is the conditional-reference repair plan.  It is an
extension/invention until tied to the cited Zhao-Cui source operations.  Its
first deliverable is an independent, parameter-dependent innovation-coordinate
value/score authority, not a promoted Zhao-Cui candidate.

## Ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Focused mechanics, reload, XLA, and memory checks passed. |
| Numerical validity | Authority reproducibility and sample-growth checks passed. |
| Scientific interpretation | Current centered representation rejected; target value/score remain to be established by the next independent reference. |

## Next decision

Implement and test the exact innovation-coordinate reference.  If its value and
total score agree across eager/compiled execution and independent seeds, use it
to audit a parent-conditioned representation.  If it does not, stop and repair
the target/model definition before any new density fitting.
