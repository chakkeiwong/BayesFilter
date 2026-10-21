# LG1 finite-endpoint comparisons, attempt 4

The finite DCTR posterior differs from the exact model posterior by more than
the frozen median-CDF margin. **D/R is DISCREPANCY**, conditional on the declared
empirical quadrature allowances. This is evidence against this finite likelihood's
accuracy on the attempt 4 dataset. It is not evidence that an imperfect DCTR
force cannot drive correctly corrected HMC. The separate C/D sampler test
completed both planned looks at 1000 and 2000 retained draws per chain. C/D and
C/R are both **INCONCLUSIVE** at the terminal limit. Five of six C/D functions
meet their equivalence margins; the remaining median-CDF interval is too wide.
B/R previously finished INCONCLUSIVE with healthy retained chains.

The target is `lg1_dctr_physical_fp32_chart_fp64_v1`: unchanged full physical
FP32 DCTR values, a bounded FP64 chart, and explicit adjacent-interior FP32
rounding at support boundaries. The uniform prior constant and chart Jacobian
appear exactly once. The proposal remains the original imperfect full DCTR
field. The numerical endpoint qualification checked physical-value non-harm,
extreme chart inputs, replay, permutation and row independence on the actual
GPU route. This named target differs from the historical unconstrained FP32
scalar, whose numerical support failures were not acceptable for this test.

D independently integrates the finite target over the complete physical support
using 128, 256, 512 and 1024 trapezoidal intervals. Every node was value-valid;
there were no force-invalid nodes. Two boundary-limit evaluations per grid
recorded coordinate rounding. The four grids passed the predeclared convergence
screen. D and R evaluate the same six frozen functions, centers, scales and
thresholds, including the exact conditional LG1 predictive probability. These
comparisons assess parameter uncertainty, not DCTR state prediction.

| D/R function | Difference | Empirical interval | Margin | Decision |
|---|---:|---|---:|---|
| Standardized phi | 0.0471310 | [0.0471125, 0.0471496] | 0.10 | Equivalent |
| Centered second moment | 0.0249845 | [0.0249461, 0.0250230] | 0.10 | Equivalent |
| CDF at R's 0.1 quantile | -0.00421163 | [-0.00422095, -0.00420230] | 0.02 | Equivalent |
| CDF at R's 0.5 quantile | -0.0211172 | [-0.0211252, -0.0211092] | 0.02 | Discrepancy |
| CDF at R's 0.9 quantile | -0.0117931 | [-0.0118132, -0.0117730] | 0.02 | Equivalent |
| Pr(y11 > 0 given y) | -0.0105823 | [-0.0105857, -0.0105789] | 0.03 | Equivalent |

These are numerical intervals formed from the sum of empirical D/R refinement
allowances, not confidence intervals or rigorous integration bounds. A grid
aliasing error could escape that screen; the four-grid consistency makes it
less plausible but does not prove a bound. The median discrepancy exceeds its
margin by about 0.001117, compared with a combined empirical allowance of
0.00000803. No controls were selected using this discrepancy.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject endpoint-accuracy claim for this scope | One of six D/R margins fails | All reference node-validity and refinement checks pass | Empirical quadrature allowance; one dataset and particle bank | Preserve the failed endpoint claim; any endpoint repair needs fresh scoped data | Rejection of surrogate HMC or of every DCTR likelihood |
| C/D remains viable but unconfirmed | Five of six equivalence intervals pass; no resolved discrepancy | All retained health, mixing and MCSE checks pass | Median-CDF interval exceeds its precision target | Close this attempt without extra looks; continue the next model scope | Full posterior equivalence or a sampler failure |
| C/R unresolved | Three of six equivalence intervals pass; no resolved discrepancy | Retained health checks pass; D/R endpoint discrepancy remains | Combined sampling and endpoint error | Keep C/R distinct from the independent D/R result | Correct model-posterior inference |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Reference and retained-chain health pass; the declared endpoint-accuracy promotion margin fails |
| Statistically supported ranking | None; this is a deterministic reference comparison with empirical numerical allowances |
| Descriptive-only differences | Runtime, acceptance, ESS and R-hat; these do not establish efficiency superiority |
| Default readiness | Not established; heuristic dominance is NOT_RUN |
| Next evidence needed | A separately planned, higher-precision replication; heuristic comparisons, independent datasets/banks and other model scopes |

The retained C archive contains four chains of2000 draws, excluding all public
tuning warmup and verification draws. Maximum rank/folded R-hat is1.000703,
bulk ESS3768 and tail ESS3130; every frozen function passes its MCSE diagnostic.
All8000 transitions pass the declared health screen. Independent archive
inspection confirms every checksum and finds zero divergences, zero force
fallbacks, zero numerical endpoint failures and zero endpoint-boundary rounding
flags. Per-chain acceptance is0.7275–0.742, an explanatory diagnostic only.

| C/D function | Difference | Jointly adjusted interval | Margin | Decision |
|---|---:|---|---:|---|
| Standardized phi | -0.0101395 | [-0.0673041,0.0470250] |0.10| Equivalent |
| Centered second moment |0.00336740| [-0.0696381,0.0763729] |0.10| Equivalent |
| CDF at R's0.1 quantile | -0.000788372 | [-0.0154264,0.0138497] |0.02| Equivalent |
| CDF at R's0.5 quantile |0.00299220| [-0.0239451,0.0299295] |0.02| Inconclusive |
| CDF at R's0.9 quantile |0.00154307| [-0.0158326,0.0189187] |0.02| Equivalent |
| Pr(y11>0 given y) |0.00216905| [-0.0101066,0.0144447] |0.03| Equivalent |

The intervals include empirical reference allowances and the predeclared joint
adjustment over six functions, two looks and both C/D and C/R. Their stochastic
interpretation assumes a stationary-chain CLT and approximately independent
batch means. Every C/D and C/R interval contains zero. The C/R median-CDF
interval is[-0.0450545,0.00880452]: sampling uncertainty does not resolve the
separate D/R discrepancy. An inconclusive interval neither proves equivalence
nor rejects the sampler. No post-hoc extension or margin change was made.

The engineering ledger passes for this checked run. The numerical/sampler
ledger passes the declared health and mixing screens. The scientific ledger
has one finite-endpoint accuracy rejection and unresolved overall sampler
equivalence. The strongest alternative explanation for the unresolved C/D
result is insufficient precision, although a small systematic sampler error
also remains possible. A predeclared independent replication with narrower
intervals would distinguish those explanations. No stochastic method ranking
or default promotion follows.

Strongest alternative explanation: unresolved fine-scale quadrature error or a
finite-arithmetic difference between a consumer and its reference. A qualified
reference that places the median difference inside the margin would overturn
the present endpoint conclusion. Endpoint identity, source checksums and exact
physical-value checks protect the declared comparison, but they do not prove
uniform numerical accuracy at every possible HMC state.

Evidence: [endpoint qualification](artifacts/surrogate-hmc-all-models-20260919-lg1-finite-endpoint-01/P2-endpoint/report.json),
[D reference and D/R result](artifacts/surrogate-hmc-all-models-20260920-lg1-finite-reference-01/P2-finite-reference/report.json),
[D run manifest](artifacts/surrogate-hmc-all-models-20260920-lg1-finite-reference-01/run-manifest.json),
and [accounting](artifacts/surrogate-hmc-all-models-20260920-lg1-finite-reference-01/accounting.json).
The manifest records the actual command, checkout, environment, scoped inputs,
plan and output paths; worker records include GPU memory growth, TF32 and XLA.
D charged 174.946 GPU seconds and 172.753 CPU-worker seconds. All earlier costs
remain charged. The [finite-comparison plan](surrogate-hmc-lg1-finite-comparisons-2026-09-20.md)
governs C; no extra A/B look or change to the frozen margins follows from D/R.

C evidence: [retained report](artifacts/surrogate-hmc-all-models-20260920-lg1-C-sample-01/P2-C-sample/report.json),
[launch manifest](artifacts/surrogate-hmc-all-models-20260920-lg1-C-sample-01/run-manifest.json),
[incremental accounting](artifacts/surrogate-hmc-all-models-20260920-lg1-C-sample-01/accounting.json),
and [independent archive inspection](artifacts/surrogate-hmc-all-models-20260920-lg1-C-inspection-01/inspection.json).
The retained launch charged2721.357GPU seconds and2605.313CPU-worker seconds
including master overhead; inspection added3.994CPU-worker seconds. All tuning,
reference and earlier failed work remains charged separately. Recompilation
between fixed-size chunks added runtime and is a performance limitation; it
did not change the frozen kernel or the predeclared evidence criteria.
