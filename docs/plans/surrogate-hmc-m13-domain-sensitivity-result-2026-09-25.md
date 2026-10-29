# M13 same-data domain sensitivity

Expanding the numerical integration region from four to five prior standard
deviations passed the declared materiality screens. All 9,344 new likelihood
evaluations were valid; 66,564 core evaluations were reused. The canonical goal
remains testing surrogate HMC. This result qualifies a reference diagnostic;
it does not substitute for posterior chains.

The fixed 4096-particle x8-replicate endpoint and the independent 16384-particle
x24-replicate comparison used the same observations (seed1704), Gaussian prior
and full-support wrapped-observation likelihood. The fine129 core was retained
and a positive coarse shell added. The maximum mean/SD change plus three paired
MCSEs was 4.74143005590791e-6 posterior SD (0.000474143%). The reference shell
probability was 1.0381558976941089e-7 with MCSE3.5481862668152444e-11, about
0.0000104% posterior probability. The maximum fixed-endpoint discrepancy
envelope remained .00036204138517318264 posterior SD, against the declared
.05 margin. No tuning parameter was chosen from these results.

Artifact directory:
`docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-domain-sensitivity-01`.
The report, grid values, quadrature weights, commands, device/memory policy,
source hashes and accounting are preserved there. The master confirms unchanged
sources. All31 focused CPU tests passed; GPU devices were intentionally hidden.
The worker used escalated GPU access, FP64 independent reference calculations,
XLA, TF32 enabled and verified memory growth. The campaign force default stays
FP32. Wall time511.9149 seconds; charge540.670963 CPU worker seconds and
517.0975791409728 GPU seconds. Cumulative51745.458323292CPU /
40914.26400198684GPU; remaining2254.541676708CPU /185885.73599801316GPU.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Domain screen passed | Changes <=.01SD; shell probability envelope<=1e-4; endpoint envelope<=.05SD | No invalid nodes; source unchanged | No rigorous bound outside5SD | Four-chain public-kernel mechanics/cost inside the same master | Converged HMC, complete R^2 tail proof, control selection |

| Inference status | Finding |
|---|---|
| Hard veto screen | No supported numerical invalidity or artifact corruption |
| Statistically supported ranking | None; no sampler ranking was attempted |
| Descriptive-only differences | Runtime and single-case performance; materiality envelopes use paired replicate influence MCSE |
| Default-readiness | No new default or sampler admission; FP32 owner directive retained |
| Next evidence needed | Native four-chain connection, scoped controls, public tuning and retained posterior comparison |

Engineering correctness: independent1D separable quadrature checks validated the
positive shell construction. Numerical validity: all nodes valid, refinement
and domain screens passed. Scientific interpretation: the endpoint differences
are practically small for this data and tested domain. The strongest alternative
explanation is a missed remote mode outside the checked domain or common bias in
the importance reference. Large-domain posterior mass or a discrepant independent
reference would overturn the interpretation; a global tail guarantee is the
weakest remaining aspect of this diagnostic. That limitation does not prevent
the next bounded HMC connection test.
