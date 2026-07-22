# GenUT Transport Repair Comparison

This report compares historical finite scalars with the repaired realized-row-quotient/terminal-balance program. Historical values are not exact oracles unless explicitly identified; differences are descriptive because the finite program changed.

## Execution

- Device: `RTX 4080 SUPER`; FP32, TF32, XLA; particle policy `N>1000`.
- Runtime score: `recursive forward sensitivity; no autodiff or finite difference`.
- Austria SIR is reported as an independent canonical fixed-SGQF value-only regression, not a GenUT score test.
- Revoked baselines are excluded: reduced SIR/J=1 and original iid-normal SV.

## Model Comparison

| Scope | Quantity | Prior | Modified | Paired delta 95% CI |
|---|---|---:|---:|---:|
| LGSSM T=2 | value | -8.85711 [-8.95436, -8.75986] | -8.85602 [-8.95241, -8.75963] | 0.00109231 [-0.0105736, 0.0127582] |
| LGSSM T=2 | phi1 | 3.89924 [3.58628, 4.21219] | 3.89227 [3.57774, 4.2068] | -0.00696908 [-0.061248, 0.0473099] |
| LGSSM T=2 | phi2 | -0.388488 [-0.471855, -0.305121] | -0.369308 [-0.439954, -0.298661] | 0.0191802 [-0.00138497, 0.0397454] |
| LGSSM T=2 | phi3 | -0.0696797 [-0.117442, -0.0219177] | -0.0737142 [-0.122311, -0.0251173] | -0.00403446 [-0.0210496, 0.0129807] |
| LGSSM T=2 | q_scale | 4.44879 [4.01875, 4.87884] | 4.42344 [4.00672, 4.84016] | -0.0253487 [-0.115153, 0.0644555] |
| LGSSM T=2 | r_scale | 11.2811 [10.6139, 11.9484] | 11.3135 [10.6432, 11.9838] | 0.0323367 [-0.0172835, 0.0819568] |
| LGSSM T=10 | value | -32.1765 [-32.2584, -32.0945] | -32.1229 [-32.1997, -32.0462] | 0.0535119 [-0.0394494, 0.146473] |
| LGSSM T=10 | phi1 | 11.8789 [11.423, 12.3348] | 11.5205 [11.0838, 11.9571] | -0.358416 [-0.795628, 0.078797] |
| LGSSM T=10 | phi2 | -0.367055 [-0.499819, -0.23429] | -0.40082 [-0.556903, -0.244737] | -0.0337652 [-0.147656, 0.0801258] |
| LGSSM T=10 | phi3 | -1.40187 [-1.55819, -1.24555] | -1.36135 [-1.46832, -1.25438] | 0.0405225 [-0.0880282, 0.169073] |
| LGSSM T=10 | q_scale | 10.1084 [9.56443, 10.6523] | 9.53993 [8.91629, 10.1636] | -0.568421 [-1.10696, -0.0298851] |
| LGSSM T=10 | r_scale | 14.7146 [14.1213, 15.3079] | 14.4166 [13.8989, 14.9344] | -0.298004 [-0.757338, 0.161329] |
| LGSSM T=50 | value | -136.085 [-136.308, -135.862] | -135.966 [-136.227, -135.705] | 0.118585 [-0.0907361, 0.327905] |
| LGSSM T=50 | phi1 | 5.5189 [4.95317, 6.08464] | 5.11394 [4.5399, 5.68799] | -0.404958 [-1.07188, 0.261967] |
| LGSSM T=50 | phi2 | -4.11845 [-4.33976, -3.89714] | -3.83099 [-4.106, -3.55598] | 0.287455 [-0.0210326, 0.595943] |
| LGSSM T=50 | phi3 | 0.314018 [0.10344, 0.524596] | 0.211606 [-0.00100844, 0.42422] | -0.102412 [-0.322247, 0.117423] |
| LGSSM T=50 | q_scale | -1.84212 [-3.16629, -0.517949] | -1.64653 [-2.76666, -0.526396] | 0.19559 [-1.27084, 1.66202] |
| LGSSM T=50 | r_scale | 5.35036 [3.75567, 6.94505] | 4.61312 [3.10256, 6.12367] | -0.737242 [-2.0626, 0.588117] |
| Fresh exact transformed SV | theta_gamma | -0.819051 [-0.878237, -0.759866] | -0.808734 [-0.868695, -0.748772] | 0.0103178 [0.00499972, 0.0156358] |
| Fresh exact transformed SV | theta_log_beta | -2.28027 [-2.39144, -2.1691] | -2.2855 [-2.39735, -2.17365] | -0.00523095 [-0.0152285, 0.00476659] |
| Fresh exact transformed SV | value | -116.805 [-116.855, -116.754] | -116.803 [-116.855, -116.75] | 0.0022397 [-0.00108189, 0.0055613] |
| Predator-prey T=20 | K | 1.1987 [1.13625, 1.26115] | 1.20001 [1.13738, 1.26264] | 0.00131445 [-0.000715933, 0.00334483] |
| Predator-prey T=20 | a | -0.00152326 [-0.00320513, 0.000158609] | -0.00142341 [-0.00314598, 0.000299163] | 9.98495e-05 [-6.71044e-05, 0.000266803] |
| Predator-prey T=20 | r | -22.1478 [-22.847, -21.4487] | -22.1078 [-22.8072, -21.4084] | 0.0400692 [-0.00450795, 0.0846464] |
| Predator-prey T=20 | s | -3.15117 [-3.41685, -2.88549] | -3.16701 [-3.44119, -2.89282] | -0.0158332 [-0.0309397, -0.000726729] |
| Predator-prey T=20 | u | -0.640241 [-1.07682, -0.203662] | -0.65504 [-1.09922, -0.210862] | -0.0147995 [-0.052188, 0.022589] |
| Predator-prey T=20 | v | 0.154255 [-0.384672, 0.693182] | 0.172897 [-0.37561, 0.721403] | 0.0186415 [-0.0279032, 0.0651862] |
| Predator-prey T=20 | value | -103.162 [-103.33, -102.993] | -103.164 [-103.333, -102.995] | -0.00204182 [-0.00670271, 0.00261907] |
| Actual Austria SIR | value | -691.36920683 | -691.36920683 | 0.000e+00 |
| Actual Austria SIR | score | not applicable | not applicable | not applicable |

Intervals are Student-t 95% intervals over particle seeds. They are descriptive estimator uncertainty intervals, not proof of equality or superiority.

## Controls

| Scope | Prior controls | Modified controls |
|---|---|---|
| LGSSM T=2 | `{'epsilon': 2.0, 'sinkhorn_steps': 8, 'balance_steps': 'not present in historical scalar', 'ridge': 1e-05}` | `{'balance_steps': 4, 'epsilon': 4.0, 'ridge': 1e-05, 'sinkhorn_steps': 4}` |
| LGSSM T=10 | `{'epsilon': 2.0, 'sinkhorn_steps': 8, 'balance_steps': 'not present in historical scalar', 'ridge': 1e-05}` | `{'balance_steps': 4, 'epsilon': 2.0, 'ridge': 1e-06, 'sinkhorn_steps': 4}` |
| LGSSM T=50 | `{'epsilon': 2.0, 'sinkhorn_steps': 8, 'balance_steps': 'not present in historical scalar', 'ridge': 1e-05}` | `{'balance_steps': 8, 'epsilon': 4.0, 'ridge': 1e-05, 'sinkhorn_steps': 4}` |
| Fresh exact transformed SV | `{'epsilon': 2.0, 'sinkhorn_steps': 4, 'balance_steps': 0, 'ridge': 1e-05}` | `{'balance_steps': 8, 'epsilon': 4.0, 'ridge': 1e-06, 'sinkhorn_steps': 4}` |
| Predator-prey T=20 | `{'epsilon': 4.0, 'ridge': 1e-06, 'sinkhorn_steps': 8}` | `{'balance_steps': 8, 'epsilon': 2.0, 'ridge': 1e-06, 'sinkhorn_steps': 4}` |
| Actual Austria SIR | `N/A` | `N/A` |

## Structural Model

Truth (physical): `[0.8, 0.5, 0.7, 0.4, 0.25]`.
Selected repaired controls: `{'balance_steps': 16, 'epsilon': 4.0, 'ridge': 1e-06, 'sinkhorn_steps': 4}`.

| Parameter | Truth | GenUT physical score / 95% CI | Same-target UKF physical score |
|---|---:|---:|---:|
| rho | 0.8 | 154.286 [-252.312, 560.884] | -18.3211 |
| sigma | 0.5 | 92.7731 [-110.808, 296.354] | -18.2111 |
| phi | 0.7 | 43.06 [-106.196, 192.316] | -20.4998 |
| gamma | 0.4 | 61.4542 [-92.9247, 215.833] | -17.0123 |
| R | 0.25 | -5.97103 [-26.3351, 14.3931] | -6.83976 |
| value | N/A | -124.16 [-124.722, -123.598] | -124.46262801 |

GenUT mean physical score: `[154.28605047929963, 92.77309511506378, 43.059977692945296, 61.45420550772917, -5.971026940010125]`.
Hard gates: `{'all_finite': True, 'maximum_reset_residual': 3.9013102650642395e-06, 'maximum_transition_residual': 9.5367431640625e-07, 'maximum_score_sum_relative_residual': 8.19349111225165e-07}`.
Interpretation: candidate included but not leaderboard-admitted; UKF is a same-target approximation diagnostic, not an oracle.

## Decision

- Hard veto screen: all modified scopes passed finite/device/residual screens.
- Statistically supported ranking: none; prior-versus-modified scalar differs by the repaired finite program.
- Default readiness: not established.
- Next evidence: independent nonlinear score authority and broader model-specific claim campaigns.
