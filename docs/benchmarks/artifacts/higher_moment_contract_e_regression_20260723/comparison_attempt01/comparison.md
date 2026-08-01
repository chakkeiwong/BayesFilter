# Higher-Moment Contract E Candidate Comparison

Hard screen: PASS. Statistical ranking: NONE. Default readiness: NOT READY.

| Scope | Quantity | Candidate mean [95% CI] | Prior mean [95% CI] | Candidate oracle error mean | Paired abs-error delta 95% CI |
|---|---|---:|---:|---:|---:|
| LGSSM T=2 | value | -8.82024 [-8.91467, -8.7258] | -8.82053 [-8.91498, -8.72609] | 0.0419149 | [-0.000430764, 0.000420512] |
| LGSSM T=2 | phi1 | 3.95554 [3.67523, 4.23584] | 3.95294 [3.6717, 4.23418] | 0.127521 | [-0.00421028, 0.00297581] |
| LGSSM T=2 | phi2 | -0.374019 [-0.431688, -0.316349] | -0.37491 [-0.43356, -0.31626] | 0.0101624 | [-0.00282266, 0.00117648] |
| LGSSM T=2 | phi3 | -0.0490514 [-0.0980155, -8.74276e-05] | -0.0493505 [-0.0982079, -0.000493026] | 0.0340358 | [-0.000384494, 0.00161302] |
| LGSSM T=2 | q_scale | 4.65774 [4.27721, 5.03827] | 4.65674 [4.27552, 5.03796] | 0.240526 | [-0.00559593, 0.00260324] |
| LGSSM T=2 | r_scale | 11.0606 [10.4346, 11.6867] | 11.0623 [10.4364, 11.6881] | -0.0774897 | [-0.00363695, 0.0050797] |
| LGSSM T=10 | value | -32.1401 [-32.2411, -32.0391] | -32.1394 [-32.2407, -32.0381] | -0.08748 | [-0.00166422, 0.000546753] |
| LGSSM T=10 | phi1 | 11.1596 [10.7652, 11.554] | 11.1596 [10.7646, 11.5546] | -0.12039 | [-0.0094336, 0.00412592] |
| LGSSM T=10 | phi2 | -0.288911 [-0.476689, -0.101134] | -0.287571 [-0.476021, -0.0991215] | 0.0151299 | [-0.00333158, 0.000304069] |
| LGSSM T=10 | phi3 | -1.33979 [-1.45392, -1.22567] | -1.34084 [-1.45489, -1.22678] | -0.034756 | [-0.00122625, 0.00181007] |
| LGSSM T=10 | q_scale | 9.68801 [9.04147, 10.3345] | 9.67916 [9.03397, 10.3244] | 0.199347 | [-0.00698703, 0.0140784] |
| LGSSM T=10 | r_scale | 14.6019 [13.8797, 15.3241] | 14.6003 [13.8758, 15.3247] | 0.533545 | [-0.00814656, 0.0030692] |
| LGSSM T=50 | value | -136.064 [-136.346, -135.782] | -136.065 [-136.346, -135.784] | 0.0119306 | [-1.4015e-05, 0.0019824] |
| LGSSM T=50 | phi1 | 5.71923 [5.33231, 6.10614] | 5.70907 [5.32096, 6.09718] | 0.0637806 | [-0.00857203, 0.00397961] |
| LGSSM T=50 | phi2 | -4.0241 [-4.31015, -3.73805] | -4.02513 [-4.31005, -3.74022] | -0.189044 | [-0.000256299, 0.00317278] |
| LGSSM T=50 | phi3 | 0.22075 [-0.013684, 0.455185] | 0.220485 [-0.0137634, 0.454734] | -0.0816115 | [-0.00089077, 0.0022906] |
| LGSSM T=50 | q_scale | -2.21452 [-3.27136, -1.15769] | -2.23526 [-3.29294, -1.17758] | -0.297346 | [-0.0139696, 0.0076143] |
| LGSSM T=50 | r_scale | 4.40299 [2.702, 6.10398] | 4.38064 [2.68266, 6.07862] | 0.0487137 | [-0.00891503, 0.020505] |
| Fresh transformed SV T=50 | value | -116.804 [-116.856, -116.752] | -116.803 [-116.855, -116.75] | -0.0049151 | [-0.000863838, 0.00106888] |
| Fresh transformed SV T=50 | theta_gamma | -0.81244 [-0.872376, -0.752503] | -0.808734 [-0.868695, -0.748772] | 0.0395889 | [-0.00229992, 0.00205636] |
| Fresh transformed SV T=50 | theta_log_beta | -2.28351 [-2.39521, -2.17181] | -2.2855 [-2.39735, -2.17365] | -0.0519383 | [-0.00194571, 0.00112205] |

A paired absolute-error interval entirely below zero would support improvement; entirely above zero would support regression. None is entirely on one side.

## Diagnostics

| Scope | Selected controls | Max skew residual | Max kurtosis residual |
|---|---|---:|---:|
| LGSSM T=2 | {'balance_steps': 8, 'epsilon': 4.0, 'higher_moment_correction_steps': 2, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.05, 'ridge': 1e-05, 'sinkhorn_steps': 8} | 0.818607 | 1.0613 |
| LGSSM T=10 | {'balance_steps': 4, 'epsilon': 2.0, 'higher_moment_correction_steps': 2, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.05, 'ridge': 1e-06, 'sinkhorn_steps': 4} | 0.610742 | 0.900343 |
| LGSSM T=50 | {'balance_steps': 4, 'epsilon': 4.0, 'higher_moment_correction_steps': 1, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.02, 'ridge': 1e-05, 'sinkhorn_steps': 4} | 0.855043 | 2.33694 |
| Fresh transformed SV T=50 | {'balance_steps': 8, 'epsilon': 4.0, 'higher_moment_correction_steps': 2, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.05, 'ridge': 1e-06, 'sinkhorn_steps': 8} | 0.581864 | 0.81603 |
| Predator-prey T=20 | {'balance_steps': 4, 'epsilon': 2.0, 'higher_moment_correction_steps': 2, 'higher_moment_floor': 1e-05, 'higher_moment_strength': 0.05, 'ridge': 1e-06, 'sinkhorn_steps': 4} | 0.426158 | 0.793381 |

## Austria SIR

- Candidate value: -691.369206826; prior value: -691.369206826; difference: 0.000e+00.
- Score is not applicable because the canonical fixed route has no free parameter.

## Nonclaims

- The recursive score is the score of the executed finite approximation, not an exact posterior score.
- Predator-prey score results are descriptive without an exact oracle.
- Nonzero moment residuals reject an exact higher-moment matching claim.
- This campaign does not promote the candidate to canonical/default/HMC/leaderboard status and says nothing about NAWM.
