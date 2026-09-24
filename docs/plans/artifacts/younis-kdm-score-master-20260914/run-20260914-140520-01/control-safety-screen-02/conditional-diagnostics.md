# Control-safety conditional diagnostics

Exploratory screen with two fresh datasets and one particle stream per condition. All score-error and between-control differences are descriptive. The reference is an independently refined FP64 numerical grid. No statistical ranking, control selection, provider promotion or default change is issued.

## weak

| Arm | Completed | Mean squared score error | Largest score change from baseline | Observed heuristic veto |
|---|---:|---:|---:|---|
| baseline | 2/2 | 4.485372 | 0 | true |
| damping_0 | 2/2 | 4.485076 | 0.0001491876 | true |
| damping_001 | 2/2 | 4.485108 | 0.0001327977 | true |
| damping_1 | 2/2 | 4.487553 | 0.001041019 | true |
| trust_1 | 2/2 | 4.486141 | 0.0003075099 | true |
| trust_2 | 2/2 | 4.485308 | 2.876389e-05 | true |
| ridge_1e7 | 2/2 | 4.485367 | 2.023049e-06 | true |
| ridge_1e3 | 2/2 | 4.486228 | 0.0003893478 | true |
| epsilon_05 | 2/2 | 4.349989 | 0.05376335 | true |
| epsilon_4 | 2/2 | 4.58349 | 0.04547701 | true |
| flow_16 | 2/2 | 5.595772 | 0.3778325 | true |
| transport_60 | 2/2 | 4.485372 | 3.371748e-07 | true |
| bootstrap | 2/2 | 2.108782 | 4.229392 | adversary / incomplete |
| local_linear | 2/2 | 0.1846445 | 3.249463 | adversary / incomplete |
| ekf | 2/2 | 0.3301111 | 3.323951 | adversary / incomplete |
| ukf | 2/2 | 0.2402934 | 3.29484 | adversary / incomplete |
| coordinate_cap_4 | 2/2 | 4.613144 | 0.09996955 | true |
| coordinate_cap_8 | 2/2 | 4.612931 | 0.1001796 | true |

## curved

| Arm | Completed | Mean squared score error | Largest score change from baseline | Observed heuristic veto |
|---|---:|---:|---:|---|
| baseline | 2/2 | 0.4880303 | 0 | false |
| coordinate_cap_4 | 2/2 | 0.674905 | 0.3508826 | true |
| coordinate_cap_8 | 2/2 | 0.6860895 | 0.3523213 | true |
| damping_0 | 2/2 | 0.4888832 | 0.004231268 | false |
| damping_001 | 2/2 | 0.4888834 | 0.003755883 | false |
| damping_1 | 2/2 | 0.4835347 | 0.01293076 | false |
| trust_1 | 2/2 | 0.4883798 | 0.002112913 | false |
| trust_2 | 2/2 | 0.4878137 | 0.0007449936 | false |
| ridge_1e7 | 2/2 | 0.4880294 | 1.019964e-05 | false |
| ridge_1e3 | 2/2 | 0.4881965 | 0.001016907 | false |
| epsilon_05 | 2/2 | 0.5196742 | 0.160439 | true |
| epsilon_4 | 2/2 | 0.535443 | 0.2041148 | true |
| flow_16 | 2/2 | 0.4612127 | 0.052609 | false |
| transport_60 | 2/2 | 0.4880307 | 6.899736e-07 | false |
| bootstrap | 2/2 | 2.834118 | 1.792433 | adversary / incomplete |
| local_linear | 2/2 | 0.5114413 | 0.8453367 | adversary / incomplete |
| ekf | 2/2 | 47.06395 | 9.133013 | adversary / incomplete |
| ukf | 2/2 | 36.57285 | 8.303561 | adversary / incomplete |

## concentrated

| Arm | Completed | Mean squared score error | Largest score change from baseline | Observed heuristic veto |
|---|---:|---:|---:|---|
| baseline | 2/2 | 5.771415 | 0 | true |
| coordinate_cap_4 | 2/2 | 6.165813 | 0.1745683 | true |
| coordinate_cap_8 | 2/2 | 6.167951 | 0.1755378 | true |
| damping_0 | 2/2 | 5.771447 | 0.0001440921 | true |
| damping_001 | 2/2 | 5.771445 | 0.0001289485 | true |
| damping_1 | 2/2 | 5.771405 | 0.0008617921 | true |
| trust_1 | 2/2 | 5.773171 | 0.0007768746 | true |
| trust_2 | 2/2 | 5.771253 | 6.99149e-05 | true |
| ridge_1e7 | 2/2 | 5.771404 | 4.352364e-06 | true |
| ridge_1e3 | 2/2 | 5.772417 | 0.0005272557 | true |
| epsilon_05 | 2/2 | 5.849653 | 0.05741881 | true |
| epsilon_4 | 2/2 | 5.916289 | 0.05267603 | true |
| flow_16 | 2/2 | 7.140148 | 0.615345 | true |
| transport_60 | 2/2 | 5.771415 | 1.36961e-06 | true |
| bootstrap | 2/2 | 3.658326 | 4.207028 | adversary / incomplete |
| local_linear | 2/2 | 1.06098 | 3.614169 | adversary / incomplete |
| ekf | 2/2 | 12.04128 | 7.516933 | adversary / incomplete |
| ukf | 2/2 | 11.91327 | 7.498002 | adversary / incomplete |

