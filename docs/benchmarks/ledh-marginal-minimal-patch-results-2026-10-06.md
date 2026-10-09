# Shared LEDH correction: direct patch and matched results

The direct patch reproduces the previous LGSSM and KSC calculations when each child uses its own component, and reproduces the other branch’s range-bearing marginal calculation when each child uses all components. No broad branch merge was needed. All reported checks use CPU FP64/XLA as an explicit mechanics/reference exception.

Only three runtime files change: the shared correction helper, the existing analytical core (to carry its affine map and tangent), and the two batch entry points in their existing module. Both Gaussian component selections use the same density and total-derivative algebra. The ancestor selection remains the default and takes O(N) density work; full mixing uses O(N²) work with exact repository chunks. Existing custom non-Gaussian and annealed ancestor routes retain their existing calculations; unsupported requests for their marginal extension fail explicitly.

For child i, let S_i={i} select its own ancestor or let S_i contain every ancestor. The unnormalized weight is

\[V_i=\omega_i g(x_i)\frac{\sum_{j\in S_i}\omega_j f_j(x_i)}{\sum_{j\in S_i}\omega_j q_j(x_i)}.\]

For S_i={i}, the inner weights cancel and the ratio is f_i/q_i; the outer weight remains. Each Gaussian proposal has mean F_j(m_j) and covariance B_j Q B_jᵀ. The existing flow loop now returns these quantities and their analytical derivatives. The total derivative includes incoming weights, moving particles, moving component means and every factor of B_j Q B_jᵀ. This changes neither the flow equations nor the UKF/reset/moment-correction formulas.

## Numerical agreement

The fixed design has N=64 particles, T=1 and T=20, and two parameter points per model. Contract E, marginal and pairwise moment corrections, and coordinate/trust caps are enabled in both checkouts. All 24 patched evaluations are finite; 16 direct compatibility comparisons meet the predeclared tolerances (value 1e-8 + relative 1e-8; score 1e-7 + relative 1e-8). Source commit: `f5e69d716818ff9ecf28b732a8952f5ec7f87739`. The same diagnostic model adapters and identical hashed inputs are used in both checkouts. This is not a replay of the previous large campaign’s particular random designs.

| Comparison | Cases | Max absolute log-likelihood difference | Max absolute score-component difference |
|---|---:|---:|---:|
| lg3, ancestor, versus pre-edit local | 4 | 2.842e-14 | 5.329e-15 |
| ksc, ancestor, versus pre-edit local | 4 | 7.105e-15 | 2.220e-15 |
| m13, ancestor, versus pre-edit local | 4 | 6.054e-12 | 1.975e-10 |
| m13, marginal_mixture, versus source f5e69d716 | 4 | 1.688e-13 | 2.203e-12 |

Actual T=20 values (each score lists both parameter derivatives):

| Model / selection / point | Reference log-likelihood | Patched log-likelihood | Reference score | Patched score |
|---|---:|---:|---|---|
| lg3 / ancestor / [1.0, 0.0] | -87.517705838215 | -87.517705838215 | (-8.638157028306, -18.343105599261) | (-8.638157028306, -18.343105599261) |
| ksc / ancestor / [0.0, -0.7] | -36.164686896102 | -36.164686896102 | (-1.099185537473, 1.151267546788) | (-1.099185537473, 1.151267546788) |
| m13 / marginal_mixture / [0.0, 0.0] | 31.423264261466 | 31.423264261466 | (14.297109423824, 10.502487509881) | (14.297109423824, 10.502487509881) |
| m13 / marginal_mixture / [-0.73272085, -0.01465508] | -9.081671234021 | -9.081671234021 | (155.841880512857, 69.084592524565) | (155.841880512857, 69.084592524567) |

Full precision values, including T=1 and the second LGSSM/KSC point, are in `docs/plans/artifacts/ledh-marginal-minimal-patch-20261006-01/comparison.json`. LGSSM uses transition-scale and log-process-scale parameters. KSC uses probit persistence and log beta. Range-bearing uses log process/observation standard-deviation scales. Initial arrays are fixed, so these are derivatives of the declared fixed-initial-input program.

## Checks and limitations

35 distinct tests pass: 23 existing batch/full-score regressions and 12 focused checks. The focused checks compare mixture densities with an independent reference, differentiate all inputs with nonuniform weights, check collapse for identical components, check recursive analytical scores by finite differences, verify both batch endpoints actually call the shared correction, check the default/unsupported-policy behavior, and exercise N=4096 with the required K=2048 two-by-two tiling. The initial test run had three fixture failures: a prohibited N=1 case and two comparisons that inherited different existing wrapper defaults. The fixtures were corrected; numerical tolerances were not relaxed. A missing type alias also caused an initial harness startup failure, repaired before baseline generation.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept local implementation parity | All 16 compatibility comparisons pass | All compared values/scores finite; focused derivative and wiring tests pass | FP32/GPU behavior and model-specific accuracy remain untested here | Use this small patch as the implementation for the separately planned PP/SIR canaries | Oracle accuracy, statistical improvement, default readiness, HMC readiness |

| Inference question | Status |
|---|---|
| Hard veto screen | No implementation veto remains in the tested FP64 fixtures |
| Statistically supported ranking | Not evaluated; deterministic compatibility, not stochastic accuracy comparison |
| Descriptive-only differences | Actual fixed-design likelihoods/scores above; different component selections generally produce different finite approximations |
| Default readiness | Existing ancestor selection retained; no scientific/default promotion |
| Next evidence needed | Target-specific PP/SIR canaries and GPU/FP32 checks before broader numerical claims |

Red-team review: matching another implementation could preserve a common error. Independent density references and finite-difference checks reduce that risk, but they do not establish filtering accuracy. The weakest evidence is transfer beyond these small FP64 fixtures. A failure with matched controls, valid inputs and a checked derivative would overturn the corresponding implementation conclusion. Pairwise skew/kurtosis correction and caps remain in the shared reset path and were enabled in the parity fixtures.

No main merge, commit, push, new HMC run or PP/SIR campaign was performed in this step. The implementation is reviewable on `sqmc-development`. The broader main synchronization and PP/SIR work remain separate next steps.
