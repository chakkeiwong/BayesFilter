# The saved kernel fits lose mixture uncertainty in the proposal

Completed 17 October 2026; documentation finalized 18 October (Hong Kong).
Plan: ../../ledh-ksc-information-diagnosis-20261017.md.
Branch ledh-refactor-with-policy-fix, HEAD
ea1018de1da55ed02a29a8d60f08a6827fce8a42 plus the manifest's source hashes.

## Finding

The exact product of each saved fitted message with the current-observation
Gaussian has variance 2.45 to 6.18 times the covariance produced by the
existing global proposal. The consumer computes exactly its documented
responsibility-averaged component precision. That precision does not include
uncertainty across component means.

This is a mathematical difference between two specified local quantities, not
a code/document discrepancy or proof of the cause of the full score error.
The comparator product uses the same approximate current Gaussian as the
consumer. It is not the exact filtering/smoothing distribution. The previous
kernel score failures remain; this diagnosis does not promote a repair.

## Derivation and actual call chain

For a mixture psi=sum pi_k N(c_k,S_k), write rho_k for anchor responsibilities
and s_k=S_k^-1(c_k-a). The consumer uses J=E_rho[S_k^-1]. Direct differentiation
gives -Hess(log psi)=J-Cov_rho(s_k). Thus J is positive definite, but it is not
the negative Hessian. All homoscedastic fitted kernels here use J=1/h^2.

For a current Gaussian N(a,S), the exact product with a homoscedastic mixture
has posterior component variance C0=(S^-1+h^-2)^-1, means
b_k=C0(S^-1 a+h^-2 c_k), and weights proportional to pi_k N(a;c_k,S+h^2).
Its total variance is C0+Var_alpha(b_k). The existing consumer uses only C0.
The manuscript includes this proposition and proof, including S>0 and the
single-component equality case.

Executable wiring checks verify build_maps calls the same guide at one global
anchor and all 32 local anchors. The diagnostic invokes the existing
FrozenMixtureBackwardGuide.evaluate and build_maps, then compares the actual
global-map moments with independently calculated formulas. It also checks the
guide's directional spatial derivative against the mixture Hessian. This is
an independent diagnostic consumer of those routines, not a new LEDH route.

## Conditional evidence

All 24 saved final kernel fits are examined, with each cluster's first already
recorded cloud. Exact future-message controls use the same clouds: 48 rows,
1,584 total anchors. There is no new fitting, bandwidth selection or score run.

| Condition | Fitted-kernel product/consumer variance, min–max | Median | Exact-message control median |
|---|---:|---:|---:|
| Ordinary | 4.7329–6.1772 | 5.7791 | 1.2978 |
| Negative future tail | 4.0551–6.0434 | 4.9288 | 1.1935 |
| Positive future tail | 2.4545–4.7748 | 2.8404 | 1.1047 |

Cheap local comparators, shown as median variances within each condition:

| Condition | No future guide | Existing kernel consumer | Moment-matched Gaussian message | Exact product of fitted mixture |
|---|---:|---:|---:|---:|
| Ordinary | 1.29594 | 0.17588 | 1.10043 | 0.98738 |
| Negative future tail | 1.38740 | 0.25202 | 1.14343 | 1.32677 |
| Positive future tail | 1.31242 | 0.25227 | 1.11591 | 0.74162 |

Each column is summarized separately; a ratio of column medians need not equal
the median paired ratio. These are descriptive local quantities, not score
rankings. The one-component identity is checked independently in the unit tests.
The negative Hessian is negative at 88/264, 114/264 and 50/264 fitted anchors
in the three conditions. An unqualified Hessian substitution is therefore
not justified. The exact-message control also has 59 negative anchors in the
negative-tail condition, illustrating that this is not solely fitting noise.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Consumer identity checked | Maximum relative identity error 1.43e-15, below 1e-9 | None | First-time, first-cloud coverage only | Preserve the existing evidence | Full canonical conformance |
| Omitted mixture uncertainty confirmed | Maximum homoscedastic variance-identity error 2.22e-16 | None | Role in downstream score error | Design exact-product moment conditioning in the shared global map | Score improvement or causal attribution |
| Reject direct Hessian substitution as an unreviewed repair | 252/792 fitted anchors have negative negative-Hessian | Positive-information assumption would fail | A different justified indefinite-curvature method remains possible | Use positive Gaussian-product moments for the next candidate | All Hessian-based methods are invalid |
| Close diagnosis | All 48 rows and source/input checks pass | No continuation veto | New candidate needs independent final data | Prepare a separate bounded repair plan | Default/HMC/KDM promotion |

## Inference status

| Evidence | Status |
|---|---|
| Hard veto screen | No numerical, source-drift, artifact or identity failure |
| Statistically supported ranking | None attempted; this is deterministic saved-fit mechanism diagnosis |
| Descriptive-only differences | Conditional variance ratios and curvature counts |
| Default-readiness | Not evaluated; no algorithm/default changed |
| Next evidence needed | General analytical moment-conditioning implementation, parity/derivative checks, fresh downstream comparison with unchanged baselines and conditional heuristics |

## Verification, budget and audit limitations

Four focused CPU tests pass in 5.57 seconds: independent score finite differences,
single-component equality, between-component variance and actual global/local
call-chain wiring. Attempt-01 passes all 48 rows in 6.013315 process-wall
seconds, using one of at most two attempts and 6.013315/180 diagnostic seconds.
The stage is now closed. CPU devices were deliberately selected before
TensorFlow import, with stable graph signatures, FP64 and XLA disabled as
a diagnostic exception. The CUDA-initialization message in the CPU log is
not GPU-stack evidence. No GPU campaign was launched.

The exact command is in attempt-01/manifest.json:
scripts/diagnose_ledh_ksc_information.py --source
docs/plans/artifacts/ledh-ksc-fitted-guides-20261017/final-02 --output
docs/plans/artifacts/ledh-ksc-information-20261017/attempt-01.
That manifest records Python/TensorFlow, Git, CPU choice, settings, all input
and source hashes, seeds, timing and result location. Rows and conditional
summaries are preserved alongside it.

MathDevMCP proves the scalar variance algebra equivalent using SymPy.
Its first implementation lookup failed because the index root was the main
file instead of the document directory. An initial operation audit also used
function names where the tool expects operation categories, producing
spurious missing-operation reports. The corrected scoped audit reports
consistent operation/term evidence and supported shape evidence, but its
proof parser remains inconclusive/unverified. Its report-boundary tool calls
for the source, scope and nonconclusion evidence supplied in this note. This is
not a MathDev certificate of the full proposition or implementation. The
explicit proof and independent numerical/call-chain tests supply the checked
evidence; remaining automated formalization limits are preserved in
mathdev-audit.json and mathdev-scoped-audit.json.

The 18 October manuscript build completes at 73 pages with no final LaTeX
warning or overfull box. The changed proof/results and next-study pages were
visually inspected; the stale next-comparison paragraph was corrected.
Masters and the active checkpoint agree with this result. Build commands,
logs, source/PDF hashes and the bounded render review are in
document-build/review.md.

## Post-run red team

The identity does not prove that insufficient proposal variance is the only,
or dominant, source of score error. Reset effects, mean shifts, importance
weight variability and the finite-particle log derivative remain alternatives.
The analysis deliberately reuses final fits, so it may generate a repair
hypothesis but cannot confirm one. An exact-product-moment candidate that
fails fresh score/heuristic comparisons would overturn the proposed repair
without invalidating the covariance identity.
