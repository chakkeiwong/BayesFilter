# Residual GenUT weight-gradient diagnosis

The residual FP32 error remains unresolved. The checked source-weight
decomposition identifies cancellation and upstream rounding, but neither tested
regrouping repairs the complete derivative. No precision candidate is installed,
no comparison bound changed, and no canonical LEDH score is established.

The [follow-up plan](filter_gradient_genut_weight_precision_followup_20260926.md)
used the same frozen 72-by-18 inputs and coefficients as 04190, with four
diagonal/four pairwise steps and `solve_gram_primal` as the diagnostic candidate.
All eight workers 04208--04215 were explicit CPU diagnostics; together they
charged 154.985868 seconds. No GPU worker was justified by these results.

| Question | Evidence | Result |
| --- | --- | --- |
| Does the final branch sum alone cause the error? | 04208 preserves exact forward records and the FP64 total derivative for original/changed graph/XLA inputs | No. At original coordinate 18, four contributions around 7.17, -17.57, 13.64 and -3.30 have cancellation ratio 673, but final summation contributes only +2.62e-6 / -1.19e-6 of total +3.52e-5 / -3.33e-5 errors. Its final-sum-only bound is 7.45e-6. |
| Do highest-precision dot products suffice? | 04209 combines the qualified dot family with Gram/triangular repairs | No. Changed graph and original XLA fail the unchanged gradient bound. |
| Is the moment pullback identity correct? | 04210: FP32/FP64, deliberately non-normalized weights, independent FP64 derivatives and finite differences | Yes in the tested primitive scope, including the weighted-centering residual. |
| Does moment regrouping repair the full derivative? | 04211: exact unchanged forward records | No. All four original/changed graph/XLA cells still fail the weight-gradient bound. |
| Can an explicit source-feature cut localize the original defect? | 04212 omits a report dependency; repaired 04213 passes FP64 chain-rule/telescoping checks but changes every forward record | No. Three of four cuts remove the original failed gradient coordinate. Its components cannot explain the untouched program. |
| Does a different derivative direction remove the defect? | 04214 has a loop-local capture error; wrapping the same triangular callable in 04215 preserves all graph forward records | No. Original graph coordinate 18 still fails: forward derivative -0.0618834496 versus FP64 -0.0619231299, error 3.96804e-5 against 2.12385e-5 allowed. Other tested graph directions pass. |
| Is the forward diagnostic XLA-compatible? | 04215 graph results are saved before the XLA failure | No. Higher-order loop derivative conversion fails; no XLA directional result or runtime remedy follows. |

For the weighted moment identity, let `m=sum(w*x)`, `y=x-m`,
`C=sum(w*y*y^T)`. With incoming symmetric covariance cotangent G and mean
cotangent g, `s=sum(w*y)` and `h=g-2G*s` give
`grad_x[i]=w[i]*(h+2G*y[i])` and
`grad_w[i]=x[i]^T*h+y[i]^T*G*y[i]`. Keeping s handles weights whose sum differs
from one. The diagnostic retains the original forward equations and combines
weight contributions before the coordinate reduction. Primitive correctness
does not remove rounding earlier in the complete nonlinear recurrence.

| Decision | Primary criterion / veto | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Reject both precision regroupings for adoption | Full independent gradient bounds fail | First upstream contribution causing the complete error remains unidentified | Preserve exact failing operands for a derivative-stage diagnostic that reproduces the original program | Numerical admission or ill-conditioning |
| Reject the feature cut as an explanation | Exact-primal and failure-preservation prerequisites fail | Changed fusion/rounding changes the witness | Do not reuse its telescoping components as untouched-program evidence | A complete error bound |
| Keep the forward diagnostic limited | Exact graph primal plus one failed direction; XLA unavailable | Several forward and reverse contributions may share a rounding source | Retain graph evidence and compiler limitation | A replacement analytical or XLA score |
| Continue independent engineering work | The local allocation is exhausted; global compute remains | Public/consumer/capacity and report gates remain open | Complete the installed loop-bound cost study and public execution repair | Whole-program completion or main merge |

Skeptical review: final-sum cancellation, a small reference coordinate, and a
correct local derivative are insufficient to label the complete problem
ill-conditioned. The failed feature cut demonstrates why decomposition must
first preserve the original witness. The forward graph result independently
rules out final reverse accumulation as the sole explanation for coordinate
18. The correction algorithm and its diagnostic gradient are retained as
unqualified FP32 computations in this scope; a claimed precision guarantee
requires a repair or a derived error outcome.
