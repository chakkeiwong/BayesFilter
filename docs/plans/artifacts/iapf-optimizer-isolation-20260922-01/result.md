# Changing the optimizer resolves some convergence failures but not guide quality

The independent bounded R optimizer passes the unchanged fit criterion in
57/64 fixed problems; the original TensorFlow projected-gradient solver passed
48/64. All 64 starts match the saved objective/gradient checks to 9.66e-16.
Final acceptance requires both R status zero and the original projected-gradient
tolerance; no R status-zero fit needed rejection by the latter. These are
selected deterministic problems, not statistical evidence that R or L-BFGS-B
is generally superior. Iteration caps match, computational work does not.

A solver change does not remove the density objective's bad incentive. The d2
terminal density fit still converges with shape residual 0.99997 and KL 38.72.
The same target without any future-message approximation gives the same result.
The independent implementation therefore supports the mathematical diagnosis:
correctly minimizing this finite-cloud density criterion can give a poor guide.

Relative shape passes 13/16 actual-target and 12/16 exact-target R fits, versus
9/16 in each target group for the original solver. Three actual-target and all
four exact-target d10 fits still hit the iteration cap. Even the one accepted
d10 relative-shape fit has KL 5.505 despite shape residual 0.002312. In d5, an
accepted exact-target fit has shape residual 1.54e-5 but KL 1.946. Small sampled
shape error is therefore not a reliable certificate of global guide geometry.
Exact values and rejection flags are in the CSV; rejected fits are not repairs.

All 16 exact KL-optimal diagonal guides are inside the original parameter box.
The good oracle guide is not excluded by those bounds in these cases. This
rules out box exclusion as the sole explanation of their poor fitted geometry;
it does not prove that the bounds never affect other optimizer trajectories.

| case | target | objective | TF accepted | R accepted | R KL by time |
|---|---|---|---:|---:|---|
| d2-s82-cloud_moments-initial_peak | actual | density_l2 | 4/4 | 4/4 | 2.422, 124.1, 0.2047, 38.72 |
| d2-s82-cloud_moments-initial_peak | actual | relative_shape | 3/4 | 4/4 | 0.1247, 0.04293, 0.198, 0.05225 |
| d2-s82-cloud_moments-initial_peak | exact | density_l2 | 4/4 | 4/4 | 2.422, 124.2, 0.03489, 38.72 |
| d2-s82-cloud_moments-initial_peak | exact | relative_shape | 4/4 | 4/4 | 0.09894, 0.04572, 0.03624, 0.05225 |
| d2-s82-log_quadratic-initial_peak | actual | density_l2 | 4/4 | 4/4 | 0.05085, 0.04024, 0.03931, 0.04364 |
| d2-s82-log_quadratic-initial_peak | actual | relative_shape | 4/4 | 4/4 | 0.04444, 0.03957, 0.03855, 0.0388 |
| d2-s82-log_quadratic-initial_peak | exact | density_l2 | 4/4 | 4/4 | 0.05027, 0.04104, 0.0348, 0.04364 |
| d2-s82-log_quadratic-initial_peak | exact | relative_shape | 4/4 | 4/4 | 0.04502, 0.0403, 0.03537, 0.0388 |
| d5-s82-log_quadratic-initial_peak | actual | density_l2 | 3/4 | 4/4 | 8.569, 1.79, 4.545, 7.184 |
| d5-s82-log_quadratic-initial_peak | actual | relative_shape | 2/4 | 4/4 | 1.866, 1.597, 0.5075, 0.4592 |
| d5-s82-log_quadratic-initial_peak | exact | density_l2 | 3/4 | 4/4 | 7.803, 2.152, 4.217, 7.184 |
| d5-s82-log_quadratic-initial_peak | exact | relative_shape | 1/4 | 4/4 | 0.7385, 1.946, 0.813, 0.4592 |
| d10-s82-log_quadratic-native | actual | density_l2 | 4/4 | 4/4 | 8.028, 6.131, 4.175, 4.036 |
| d10-s82-log_quadratic-native | actual | relative_shape | 0/4 | 1/4 | 15.28, 11.12, 5.505, 8.748 |
| d10-s82-log_quadratic-native | exact | density_l2 | 4/4 | 4/4 | 8.966, 6.309, 3.269, 4.036 |
| d10-s82-log_quadratic-native | exact | relative_shape | 0/4 | 0/4 | 19.89, 13.85, 9.322, 9.171 |

## Interpretation and next discriminating check

There are at least two remaining issues beyond solver speed: the diagonal
log-quadratic initializer omits cross-coordinate terms in the exact Gaussian
log target, and the normalized density criterion weights only the sampled
cloud. It may emphasize a few points and leave poor behavior elsewhere weakly
penalized. These are hypotheses, not yet established causal explanations.
Next recover the full quadratic on the same exact-target clouds, explicitly
measure the omitted-term projection, and inspect target-weight concentration
and the corresponding local normalized-density geometry. Use this to decide
whether an initialization/criterion repair is justified before implementing
another runtime solver or expanding iteration budgets.

The unchanged cloud Gaussian, strict QR, KL-diagonal and full Gaussian
comparators remain in the linked preceding records. No fixed-cloud result
clears the existing conditional filtering heuristic vetoes. The R optimizer
is an independent diagnostic and is not identified as the author's solver.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close optimizer comparison | Initial identities and original fit gate checked | Seven R cells remain rejected | Selected small problems; different optimizer paths | Preserve all minima and counts | General optimizer ranking |
| Reject optimizer-only repair | Converged fits still have poor exact-message KL | Filtering vetoes remain | Empirical criterion and diagonal initialization | Same-cloud representation/concentration diagnosis | Correct convergence establishes guide quality |

| Inference status | Finding |
|---|---|
| Hard veto screen | Input identities and finite checks pass; seven optimizer nonconvergences preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Convergence counts, objective, shape, KL, work counts and time |
| Default readiness | No production change or promotion |
| Next evidence needed | Representation and empirical-geometry checks, then fresh downstream validation of a concrete repair |

Terminal skeptical review PASS for diagnosis. Strongest alternative: minima
may differ because line searches, memory and work differ; the result makes no
runtime ranking and the same final projected-gradient criterion is checked.
Weakest evidence: only four selected small cases, and tiny sampled residuals
are not global shape guarantees. An input/unit mismatch or an oracle outside
the box would change the diagnosis; both checks passed. Unknown author choices,
paper-model timing/controller, TF32, model-score, LEDH and HMC gaps remain.

The command, exact controls and installed official R optim documentation are
preserved. R 4.1.2 ran CPU-only with GPU hidden and one BLAS/OMP thread. No
new random draws, production changes, dependencies or environment changes.
`fits.rds` preserves all parameters and optimizer messages; CSV preserves
metrics, failures and function/gradient counts. Source/input identities, budget
and terminal checks are in the manifest and verification files.
