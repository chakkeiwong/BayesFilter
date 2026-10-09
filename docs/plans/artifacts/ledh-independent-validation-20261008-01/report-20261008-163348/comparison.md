# Independent T=50 validation of the frozen covariance-guided mixture

The settings from the matched comparison were frozen before these new observations and particle designs. Each dataset has eight paired designs at N=1008, T=50, FP64 GPU/XLA. The raw likelihood and every score coordinate are in the linked CSV; reference fits and uncertainty are preserved in the JSON.

Campaign complete: **False**. Default promotion: **not evaluated**.
Heuristic comparison: `descriptive_losses_present_promotion_veto`.

The new method has larger observed errors than simpler comparators in some model/dataset/metric combinations. These observations do not support universal non-deterioration. These losses block promotion; they do not establish that the method is worse in every setting.

| Model | Data seed | Reference | Method | Mean log likelihood | Mean absolute likelihood error | Mean score L2 error | Error of mean score |
|---|---:|---|---|---:|---:|---:|---:|
| lgssm | 26100831 | model_reference | old | -138.361379 | 0.027987 | 0.104756 | 0.007578 |
| lgssm | 26100831 | model_reference | old_covariance_only | -138.361242 | 0.027962 | 0.104865 | 0.009078 |
| lgssm | 26100831 | model_reference | new | -138.332676 | 0.047157 | 0.096283 | 0.029790 |
| lgssm | 26100832 | model_reference | old | -120.111796 | 0.024913 | 0.086007 | 0.013569 |
| lgssm | 26100832 | model_reference | old_covariance_only | -120.111736 | 0.024908 | 0.085986 | 0.014116 |
| lgssm | 26100832 | model_reference | new | -120.085168 | 0.046382 | 0.093887 | 0.034927 |
| ksc | 26100831 | model_reference | old | -113.300322 | 1.653558 | 1.369230 | 1.367188 |
| ksc | 26100831 | model_reference | old_covariance_only | -113.293815 | 1.648086 | 1.358657 | 1.355194 |
| ksc | 26100831 | model_reference | new | -112.080705 | 0.150613 | 0.117371 | 0.092221 |
| ksc | 26100832 | model_reference | old | -118.207316 | 1.106189 | 1.098802 | 1.095662 |
| ksc | 26100832 | model_reference | old_covariance_only | -118.162383 | 1.081316 | 1.111285 | 1.109031 |
| ksc | 26100832 | model_reference | new | -117.205552 | 0.185479 | 0.162100 | 0.106687 |
| predator_prey | 26100831 | bootstrap_N131072 | old | -267.364613 | 0.075275 | 0.564018 | 0.286775 |
| predator_prey | 26100831 | bootstrap_N131072 | old_covariance_only | -267.364670 | 0.071228 | 0.572526 | 0.385356 |
| predator_prey | 26100831 | bootstrap_N131072 | new | -267.353725 | 0.062841 | 0.830933 | 0.144444 |
| predator_prey | 26100831 | zhao_rank20 | old | -267.364613 | 0.073403 | 0.643835 | 0.405118 |
| predator_prey | 26100831 | zhao_rank20 | old_covariance_only | -267.364670 | 0.069356 | 0.688778 | 0.545801 |
| predator_prey | 26100831 | zhao_rank20 | new | -267.353725 | 0.061615 | 0.827193 | 0.142772 |
| predator_prey | 26100832 | bootstrap_N131072 | old | -248.219561 | 0.135874 | 0.946566 | 0.690889 |
| predator_prey | 26100832 | bootstrap_N131072 | old_covariance_only | -248.215557 | 0.131625 | 0.967000 | 0.776264 |
| predator_prey | 26100832 | bootstrap_N131072 | new | -248.120605 | 0.123972 | 0.577294 | 0.316104 |
| predator_prey | 26100832 | zhao_rank20 | old | -248.219561 | 0.134680 | 0.816316 | 0.416385 |
| predator_prey | 26100832 | zhao_rank20 | old_covariance_only | -248.215557 | 0.130430 | 0.826660 | 0.502039 |
| predator_prey | 26100832 | zhao_rank20 | new | -248.120605 | 0.123375 | 0.650316 | 0.487377 |
| sir_d18 | 26100831 | bootstrap_N131072 | old | -1951.010915 | 235.543851 | 16250.053775 | 13207.310073 |
| sir_d18 | 26100831 | bootstrap_N131072 | old_covariance_only | -1969.380426 | 253.913361 | 7976.199708 | 3930.833944 |
| sir_d18 | 26100831 | bootstrap_N131072 | new | -1716.402567 | 1.183172 | 105.845035 | 103.339934 |
| sir_d18 | 26100832 | bootstrap_N131072 | old | -1837.583973 | 148.139852 | 14197.567263 | 7757.465227 |
| sir_d18 | 26100832 | bootstrap_N131072 | old_covariance_only | -1848.302316 | 158.858194 | 7315.017623 | 4715.497166 |
| sir_d18 | 26100832 | bootstrap_N131072 | new | -1689.460367 | 0.544557 | 50.212619 | 40.543708 |

Negative paired differences favor the new method. These intervals vary particle designs conditional on one dataset and a fixed numerical reference. They exclude reference bias.

| Model | Data seed | Reference | Error | Paired new minus old | Conditional 95% interval |
|---|---:|---|---|---:|---|
| lgssm | 26100831 | model_reference | absolute_value_error | 0.019170 | [-0.010356, 0.048696] |
| lgssm | 26100831 | model_reference | score_l2_error | -0.008472 | [-0.044405, 0.027460] |
| lgssm | 26100832 | model_reference | absolute_value_error | 0.021469 | [-0.025917, 0.068855] |
| lgssm | 26100832 | model_reference | score_l2_error | 0.007880 | [-0.022631, 0.038391] |
| ksc | 26100831 | model_reference | absolute_value_error | -1.502945 | [-2.350914, -0.654975] |
| ksc | 26100831 | model_reference | score_l2_error | -1.251860 | [-1.794456, -0.709263] |
| ksc | 26100832 | model_reference | absolute_value_error | -0.920710 | [-1.449450, -0.391970] |
| ksc | 26100832 | model_reference | score_l2_error | -0.936702 | [-1.234986, -0.638419] |
| predator_prey | 26100831 | bootstrap_N131072 | absolute_value_error | -0.012434 | [-0.070641, 0.045773] |
| predator_prey | 26100831 | bootstrap_N131072 | score_l2_error | 0.266915 | [-0.092578, 0.626408] |
| predator_prey | 26100831 | zhao_rank20 | absolute_value_error | -0.011788 | [-0.069395, 0.045819] |
| predator_prey | 26100831 | zhao_rank20 | score_l2_error | 0.183357 | [-0.255942, 0.622657] |
| predator_prey | 26100832 | bootstrap_N131072 | absolute_value_error | -0.011902 | [-0.103984, 0.080181] |
| predator_prey | 26100832 | bootstrap_N131072 | score_l2_error | -0.369273 | [-0.994371, 0.255825] |
| predator_prey | 26100832 | zhao_rank20 | absolute_value_error | -0.011305 | [-0.102890, 0.080280] |
| predator_prey | 26100832 | zhao_rank20 | score_l2_error | -0.166001 | [-0.815344, 0.483343] |
| sir_d18 | 26100831 | bootstrap_N131072 | absolute_value_error | -234.360679 | [-292.634726, -176.086632] |
| sir_d18 | 26100831 | bootstrap_N131072 | score_l2_error | -16144.208741 | [-35912.327532, 3623.910050] |
| sir_d18 | 26100832 | bootstrap_N131072 | absolute_value_error | -147.595294 | [-213.231652, -81.958936] |
| sir_d18 | 26100832 | bootstrap_N131072 | score_l2_error | -14147.354644 | [-27585.876134, -708.833154] |

## Actual likelihood and score values

Filter entries are arithmetic means over eight designs; their standard errors measure design variability. Reference entries pool likelihood estimates before taking logs and weight scores by likelihood. Reference standard errors exclude approximation bias. The exact/refined reference has no Monte Carlo standard error.

| Model | Data seed | Method/reference | Log likelihood | SE | Score vector | Score SE |
|---|---:|---|---:|---:|---|---|
| lgssm | 26100831 | coordinate order | | | phi_coordinate, log_q_scale, log_r_scale, initial_mean_scale | |
| lgssm | 26100831 | old | -138.361379 | 0.012045 | [5.873475, 7.324325, 4.134495, 0.099906] | [0.037524, 0.013328, 0.015804, 0.000487] |
| lgssm | 26100831 | old_covariance_only | -138.361242 | 0.012059 | [5.875246, 7.324061, 4.134517, 0.099906] | [0.037645, 0.013345, 0.015792, 0.000487] |
| lgssm | 26100831 | new | -138.332676 | 0.018412 | [5.887702, 7.340704, 4.116415, 0.099966] | [0.033678, 0.011397, 0.011267, 0.000764] |
| lgssm | 26100831 | model_reference | -138.358241 | 0.000000 | [5.867315, 7.323853, 4.130117, 0.099617] | [0.000000, 0.000000, 0.000000, 0.000000] |
| lgssm | 26100832 | coordinate order | | | phi_coordinate, log_q_scale, log_r_scale, initial_mean_scale | |
| lgssm | 26100832 | old | -120.111796 | 0.012480 | [-0.378521, -4.092488, -2.706770, 0.175238] | [0.032672, 0.009517, 0.013640, 0.000508] |
| lgssm | 26100832 | old_covariance_only | -120.111736 | 0.012517 | [-0.377945, -4.092569, -2.706815, 0.175238] | [0.032697, 0.009532, 0.013680, 0.000508] |
| lgssm | 26100832 | new | -120.085168 | 0.022626 | [-0.393613, -4.074675, -2.741511, 0.175353] | [0.030945, 0.011364, 0.013615, 0.000611] |
| lgssm | 26100832 | model_reference | -120.111047 | 0.000000 | [-0.391426, -4.090786, -2.710599, 0.175391] | [0.000000, 0.000000, 0.000000, 0.000000] |
| ksc | 26100831 | coordinate order | | | Phi_inverse_gamma, log_beta | |
| ksc | 26100831 | old | -113.300322 | 0.515770 | [-3.369361, -0.858346] | [0.233038, 0.025702] |
| ksc | 26100831 | old_covariance_only | -113.293815 | 0.511749 | [-3.354014, -0.925290] | [0.230461, 0.023793] |
| ksc | 26100831 | new | -112.080705 | 0.055023 | [-1.987706, -0.733113] | [0.027817, 0.016093] |
| ksc | 26100831 | model_reference | -111.974605 | 0.000000 | [-2.002601, -0.824124] | [0.000000, 0.000000] |
| ksc | 26100832 | coordinate order | | | Phi_inverse_gamma, log_beta | |
| ksc | 26100832 | old | -118.207316 | 0.286441 | [-1.301139, 0.457813] | [0.144098, 0.023018] |
| ksc | 26100832 | old_covariance_only | -118.162383 | 0.293229 | [-1.314746, 0.448616] | [0.142941, 0.020917] |
| ksc | 26100832 | new | -117.205552 | 0.072336 | [-0.108666, 0.381432] | [0.048407, 0.025590] |
| ksc | 26100832 | model_reference | -117.213741 | 0.000000 | [-0.205963, 0.425198] | [0.000000, 0.000000] |
| predator_prey | 26100831 | coordinate order | | | r, carrying_capacity, half_saturation, s, u, v | |
| predator_prey | 26100831 | old | -267.364613 | 0.030080 | [-34.136817, -0.784716, -0.003102, 4.151405, -4.707649, 5.155826] | [0.143219, 0.009984, 0.000402, 0.058862, 0.103847, 0.126220] |
| predator_prey | 26100831 | old_covariance_only | -267.364670 | 0.028673 | [-34.122843, -0.779869, -0.002657, 4.117557, -4.810580, 5.281467] | [0.140492, 0.009176, 0.000339, 0.055474, 0.094709, 0.114623] |
| predator_prey | 26100831 | new | -267.353725 | 0.024829 | [-33.990218, -0.776088, -0.003658, 4.134292, -4.549126, 4.959841] | [0.139019, 0.010089, 0.000831, 0.068244, 0.203115, 0.250482] |
| predator_prey | 26100831 | bootstrap_N1008 | -267.689755 | 0.150545 | [-34.344427, -1.193405, -0.004545, 4.909560, -4.760466, 5.254916] | [3.305631, 0.183828, 0.004961, 1.527168, 1.283402, 1.526186] |
| predator_prey | 26100831 | bootstrap_N32768 | -267.328636 | 0.030740 | [-34.435133, -0.782362, -0.002969, 4.034958, -4.714823, 5.155651] | [1.616286, 0.023852, 0.002806, 0.423989, 0.480274, 0.585955] |
| predator_prey | 26100831 | bootstrap_N131072 | -267.313699 | 0.021823 | [-33.893204, -0.782343, -0.003296, 4.115885, -4.611189, 5.044826] | [0.268423, 0.019891, 0.000477, 0.038154, 0.103319, 0.126382] |
| predator_prey | 26100831 | zhao_rank20 | -267.317443 | 0.000005 | [-33.991138, -0.769203, -0.003692, 4.043382, -4.478284, 4.875862] | [0.061544, 0.002449, 0.000034, 0.018866, 0.009206, 0.009770] |
| predator_prey | 26100832 | coordinate order | | | r, carrying_capacity, half_saturation, s, u, v | |
| predator_prey | 26100832 | old | -248.219561 | 0.039634 | [-10.137312, 1.487902, 0.055309, -3.199616, -12.312214, 15.362078] | [0.055119, 0.006288, 0.000937, 0.055074, 0.231491, 0.282175] |
| predator_prey | 26100832 | old_covariance_only | -248.215557 | 0.041198 | [-10.153742, 1.485433, 0.055452, -3.176665, -12.364202, 15.426173] | [0.052396, 0.007171, 0.000924, 0.056343, 0.226516, 0.275925] |
| predator_prey | 26100832 | new | -248.120605 | 0.051621 | [-10.066169, 1.486327, 0.053256, -3.196399, -11.743198, 14.668296] | [0.080833, 0.007219, 0.000472, 0.056546, 0.114427, 0.139410] |
| predator_prey | 26100832 | bootstrap_N1008 | -248.485271 | 0.343882 | [-18.139578, 1.408252, 0.040156, -2.670208, -8.575922, 10.802151] | [4.194263, 0.045838, 0.001410, 1.146172, 0.589874, 0.726165] |
| predator_prey | 26100832 | bootstrap_N32768 | -248.040382 | 0.056939 | [-8.875942, 1.525425, 0.055686, -3.279837, -12.335296, 15.389807] | [0.538516, 0.050906, 0.001336, 0.192645, 0.246528, 0.303720] |
| predator_prey | 26100832 | bootstrap_N131072 | -248.103963 | 0.022265 | [-10.158635, 1.499136, 0.054118, -3.388335, -11.894691, 14.845546] | [0.316681, 0.009544, 0.000468, 0.085016, 0.117870, 0.142065] |
| predator_prey | 26100832 | zhao_rank20 | -248.106351 | 0.000001 | [-10.105887, 1.488537, 0.054450, -3.238567, -12.050537, 15.042084] | [0.068734, 0.002432, 0.000039, 0.005739, 0.000865, 0.002251] |
| sir_d18 | 26100831 | coordinate order | | | log_kappa_scale, log_nu_scale, log_observation_noise_scale | |
| sir_d18 | 26100831 | old | -1951.010915 | 24.771149 | [-12044.317992, 5653.648006, 632.048686] | [8359.402839, 2956.768324, 1996.005554] |
| sir_d18 | 26100831 | old_covariance_only | -1969.380426 | 22.895017 | [-2701.878530, 1329.108265, 2709.570292] | [2899.912610, 854.667374, 2422.436475] |
| sir_d18 | 26100831 | new | -1716.402567 | 0.332430 | [-180.119980, 115.019125, 62.741410] | [14.152247, 10.298180, 3.540725] |
| sir_d18 | 26100831 | bootstrap_N1008 | -1716.741259 | 0.352517 | [-105.592724, 58.599784, 61.684500] | [162.083862, 52.810269, 2.448525] |
| sir_d18 | 26100831 | bootstrap_N32768 | -1715.433143 | 0.418764 | [-164.795727, 109.658299, 55.277112] | [127.498439, 39.868777, 2.172168] |
| sir_d18 | 26100831 | bootstrap_N131072 | -1715.467064 | 0.128326 | [-80.284536, 89.195049, 56.022546] | [36.570085, 10.174775, 0.507036] |
| sir_d18 | 26100832 | coordinate order | | | log_kappa_scale, log_nu_scale, log_observation_noise_scale | |
| sir_d18 | 26100832 | old | -1837.583973 | 27.762514 | [-6983.316828, 3293.931830, 1603.695977] | [6747.540166, 2256.587017, 1351.082195] |
| sir_d18 | 26100832 | old_covariance_only | -1848.302316 | 27.100457 | [-4310.456473, 2079.079723, -741.638457] | [2228.871212, 827.826392, 1848.162625] |
| sir_d18 | 26100832 | new | -1689.460367 | 0.242208 | [-158.052407, 56.949015, 8.804497] | [12.956129, 6.433033, 4.023912] |
| sir_d18 | 26100832 | bootstrap_N1008 | -1690.346970 | 0.670400 | [154.864963, -152.965097, 9.334811] | [279.957881, 163.503116, 3.712504] |
| sir_d18 | 26100832 | bootstrap_N32768 | -1689.335269 | 0.131868 | [-98.909372, 48.076780, 6.036977] | [48.166338, 26.317193, 0.512364] |
| sir_d18 | 26100832 | bootstrap_N131072 | -1689.444122 | 0.088278 | [-117.609415, 55.185253, 6.558203] | [23.621602, 4.995656, 0.486741] |

## Reference stability

Two author fits cannot bound shared TT rank or support bias. Within-fit path jackknife uncertainty and between-fit uncertainty answer different questions. The table displays both regression radii rather than selecting the radius after seeing agreement. The smaller radius supplies the displayed pooled reference. ESS and regression residuals are explanatory diagnostics, not evidence of score correctness.

| Model | Data seed | Fit seed | Radius | Likelihood | Path ESS | Maximum weight | Heldout max residual | Score vector | Conditional score SE |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| predator_prey | 26100831 | 41 | 0.000625 | -267.317448 | 99999.4 | 1.06423e-05 | 2.91681e-06 | [-33.932538, -0.766749, -0.003638, 4.023807, -4.468500, 4.865999] | [0.056002, 0.005912, 0.000116, 0.025473, 0.025574, 0.030932] |
| predator_prey | 26100831 | 41 | 0.0003125 | -267.317448 | 99999.4 | 1.06423e-05 | 3.64176e-07 | [-33.929594, -0.766754, -0.003658, 4.024516, -4.469079, 4.866092] | [0.056020, 0.005911, 0.000116, 0.025451, 0.025563, 0.030904] |
| predator_prey | 26100831 | 53 | 0.000625 | -267.317438 | 99999.3 | 1.12247e-05 | 3.30875e-06 | [-34.055697, -0.771639, -0.003706, 4.061501, -4.486890, 4.885729] | [0.052234, 0.004143, 0.000091, 0.021170, 0.020747, 0.025444] |
| predator_prey | 26100831 | 53 | 0.0003125 | -267.317438 | 99999.3 | 1.12247e-05 | 4.13554e-07 | [-34.052681, -0.771651, -0.003726, 4.062247, -4.487490, 4.885632] | [0.052294, 0.004143, 0.000091, 0.021155, 0.020759, 0.025536] |
| predator_prey | 26100832 | 41 | 0.000625 | -248.106350 | 99999.2 | 1.06348e-05 | 2.88052e-06 | [-10.040103, 1.490974, 0.054509, -3.245023, -12.050866, 15.044303] | [0.054918, 0.005914, 0.000107, 0.023449, 0.022754, 0.027770] |
| predator_prey | 26100832 | 41 | 0.0003125 | -248.106350 | 99999.2 | 1.06348e-05 | 3.59467e-07 | [-10.037153, 1.490969, 0.054488, -3.244306, -12.051402, 15.044334] | [0.054933, 0.005913, 0.000107, 0.023422, 0.022746, 0.027724] |
| predator_prey | 26100832 | 53 | 0.000625 | -248.106352 | 99998.3 | 1.08552e-05 | 3.33157e-06 | [-10.177642, 1.486117, 0.054431, -3.233572, -12.049034, 15.039884] | [0.053774, 0.004162, 0.000093, 0.022264, 0.020645, 0.025236] |
| predator_prey | 26100832 | 53 | 0.0003125 | -248.106352 | 99998.3 | 1.08552e-05 | 4.16369e-07 | [-10.174621, 1.486105, 0.054411, -3.232827, -12.049672, 15.039833] | [0.053832, 0.004162, 0.000092, 0.022273, 0.020649, 0.025311] |

| Model | Data seed | Zhao minus bootstrap log likelihood | Zhao minus bootstrap score vector |
|---|---:|---:|---|
| predator_prey | 26100831 | -0.003745 | [-0.097933, 0.013140, -0.000396, -0.072503, 0.132904, -0.168964] |
| predator_prey | 26100832 | -0.002388 | [0.052748, -0.010599, 0.000331, 0.149768, -0.155846, 0.196538] |

## Variation across the two datasets

These intervals use only two independent dataset-level error differences (df=1). They exclude reference bias and should not be read as broad cross-regime evidence.

| Model | Reference | Error | Mean new minus old | Dataset-level 95% interval |
|---|---|---|---:|---|
| lgssm | model_reference | absolute_value_error | 0.020319 | [0.005715, 0.034924] |
| lgssm | model_reference | score_l2_error | -0.000296 | [-0.104184, 0.103592] |
| ksc | model_reference | absolute_value_error | -1.211828 | [-4.910823, 2.487168] |
| ksc | model_reference | score_l2_error | -1.094281 | [-3.096507, 0.907945] |
| predator_prey | bootstrap_N131072 | absolute_value_error | -0.012168 | [-0.015551, -0.008785] |
| predator_prey | bootstrap_N131072 | score_l2_error | -0.051179 | [-4.092943, 3.990585] |
| predator_prey | zhao_rank20 | absolute_value_error | -0.011546 | [-0.014616, -0.008477] |
| predator_prey | zhao_rank20 | score_l2_error | 0.008678 | [-2.210829, 2.228186] |
| sir_d18 | bootstrap_N131072 | absolute_value_error | -190.977987 | [-742.207358, 360.251385] |
| sir_d18 | bootstrap_N131072 | score_l2_error | -15145.781692 | [-27832.000185, -2459.563200] |

## Decision and uncertainty

| Item | Finding |
|---|---|
| Completion | False |
| Primary criterion | Continuous likelihood and score errors reported per model and dataset; descriptive_losses_present_promotion_veto |
| Veto diagnostics | 0 invalid rows; 0 failed attempts; 2 running attempts; 2 missing-evidence findings |
| Main uncertainty | Two datasets; finite particle, TT rank/support, and regression-radius bias |
| Next justified action | Inspect conditional failures and reference stability; preserve frozen settings |
| Not concluded | Universal improvement, exact nonlinear oracle, default readiness or HMC validity |

| Inference item | Status |
|---|---|
| Hard veto screen | See explicit invalid/failed/missing records in comparison.json |
| Statistically supported ranking | none claimed across data-generating regimes; reference bias not bounded for nonlinear models |
| Descriptive-only differences | raw differences, per-design means, two-dataset trends, nonlinear reference differences |
| Default readiness | not evaluated or promoted |
| Next evidence needed | more independent datasets and nonlinear reference rank/path convergence before a broad ranking |

The strongest alternative explanation is favorable or unfavorable data-specific geometry combined with shared reference approximation error. Agreement of two fits alone cannot exclude shared rank or support bias. Independent-data reversals or a stable, more accurate reference would overturn a favorable interpretation. The weakest inferential element is the small number of datasets.

Artifacts: `docs/plans/artifacts/ledh-independent-validation-20261008-01/report-20261008-163348/comparison.json` and `values-and-scores.csv`. 
Aggregate worker time: 1.288 hours. 
Plan: `docs/plans/ledh-independent-validation-20261008.md`.
