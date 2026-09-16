# A08 downstream filtering diagnostic tables

Primary values are equal-sequence averages of normalized MSE, not pooled time-point averages. Runtime and evidence diagnostics are descriptive.
All methods use the same sequence set within a dimension: all 12 d1 sequences; 11 d4 sequences excluding failed guide sequence 8. The d4 table is conditional on completion and cannot support population ranking. All available outcomes, including the two unguided methods on sequence 8, remain in report.json.

| d | method | all | near zero | ordinary | large | median fit s | median total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | transition | 0.0021600 | 0.0030579 | 0.0011964 | 0.0020561 | 0.001 | 0.530 |
| 1 | stationary_prior | 0.0051131 | 0.0056793 | 0.0018754 | 0.0098854 | 0.000 | 0.673 |
| 1 | sgqf_gaussian | 0.0037369 | 0.0049775 | 0.0027870 | 0.0030887 | 0.000 | 0.564 |
| 1 | sgqf_joint | 0.0018482 | 0.0024519 | 0.0014518 | 0.0011200 | 0.202 | 0.781 |
| 1 | tt_predictive | 0.0015124 | 0.0019170 | 0.0012909 | 0.0007813 | 4.361 | 5.447 |
| 1 | tt_guided | 0.0016963 | 0.0018042 | 0.0013399 | 0.0017197 | 4.306 | 5.417 |
| 1 | tt_pair_block | 0.0016075 | 0.0017946 | 0.0012691 | 0.0015217 | 2.946 | 4.307 |
| 1 | tt_sgqf_safeguard | 0.0016702 | 0.0019506 | 0.0014199 | 0.0013987 | 6.493 | 7.713 |
| 4 | transition | 0.0066377 | 0.0093457 | 0.0041904 | 0.0065428 | 0.001 | 0.605 |
| 4 | stationary_prior | 0.0212687 | 0.0279851 | 0.0143865 | 0.0221852 | 0.000 | 0.740 |
| 4 | sgqf_gaussian | 0.0060415 | 0.0086580 | 0.0041741 | 0.0039220 | 0.000 | 0.609 |
| 4 | sgqf_joint | 0.0029703 | 0.0034087 | 0.0022337 | 0.0033048 | 0.237 | 0.852 |
| 4 | tt_predictive | 0.0072739 | 0.0105720 | 0.0052669 | 0.0038474 | 18.816 | 20.852 |
| 4 | tt_guided | 0.0066699 | 0.0090306 | 0.0053657 | 0.0034635 | 18.190 | 20.174 |
| 4 | tt_pair_block | 0.0026151 | 0.0032617 | 0.0021797 | 0.0020670 | 30.362 | 33.283 |
| 4 | tt_sgqf_safeguard | 0.0023614 | 0.0029321 | 0.0019547 | 0.0018882 | 53.885 | 56.712 |

ESS is computed before resampling from actual particle importance weights; N=512. Means use the same matched sequence sets as the MSE table. These are descriptive statistics, not target-row fitting ESS.

| d | method | mean ESS | mean ESS/N | minimum ESS | largest weight | resampling rate | mean unique ancestors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | transition | 323.25 | 0.6313 | 34.87 | 0.0581 | 0.2885 | 440.43 |
| 1 | stationary_prior | 256.79 | 0.5015 | 1.74 | 0.7555 | 0.4792 | 390.04 |
| 1 | sgqf_gaussian | 317.91 | 0.6209 | 35.80 | 0.1511 | 0.2781 | 461.26 |
| 1 | sgqf_joint | 376.40 | 0.7352 | 118.62 | 0.0538 | 0.0844 | 496.02 |
| 1 | tt_predictive | 370.31 | 0.7233 | 111.01 | 0.0526 | 0.0948 | 492.53 |
| 1 | tt_guided | 378.73 | 0.7397 | 45.56 | 0.1372 | 0.0781 | 496.66 |
| 1 | tt_pair_block | 379.67 | 0.7415 | 76.76 | 0.0872 | 0.0771 | 497.06 |
| 1 | tt_sgqf_safeguard | 375.82 | 0.7340 | 55.98 | 0.1064 | 0.0750 | 497.34 |
| 4 | transition | 175.78 | 0.3433 | 2.70 | 0.5988 | 0.7989 | 290.39 |
| 4 | stationary_prior | 75.60 | 0.1477 | 1.11 | 0.9489 | 0.9864 | 181.48 |
| 4 | sgqf_gaussian | 203.89 | 0.3982 | 12.41 | 0.2710 | 0.8625 | 326.63 |
| 4 | sgqf_joint | 311.47 | 0.6083 | 3.16 | 0.5548 | 0.3023 | 450.54 |
| 4 | tt_predictive | 192.13 | 0.3752 | 2.69 | 0.6083 | 0.8705 | 320.31 |
| 4 | tt_guided | 199.52 | 0.3897 | 14.34 | 0.2186 | 0.8750 | 329.07 |
| 4 | tt_pair_block | 310.55 | 0.6065 | 38.77 | 0.1327 | 0.3000 | 449.77 |
| 4 | tt_sgqf_safeguard | 317.53 | 0.6202 | 44.39 | 0.1254 | 0.2750 | 455.72 |

| d | heuristic | regime | paired delta | simultaneous lower | upper | coverage | precise |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | transition | all | -0.0004898 | -0.0010781 | 0.0000985 | 12 sequences, 240 cells | True |
| 1 | transition | near_zero | -0.0011073 | -0.0019716 | -0.0002431 | 12 sequences, 98 cells | True |
| 1 | transition | ordinary | 0.0002234 | -0.0001866 | 0.0006335 | 12 sequences, 110 cells | True |
| 1 | transition | large | -0.0006575 | -0.0019329 | 0.0006180 | 12 sequences, 32 cells | True |
| 1 | stationary_prior | all | -0.0034429 | -0.0058396 | -0.0010462 | 12 sequences, 240 cells | True |
| 1 | stationary_prior | near_zero | -0.0037287 | -0.0061183 | -0.0013391 | 12 sequences, 98 cells | True |
| 1 | stationary_prior | ordinary | -0.0004555 | -0.0008909 | -0.0000201 | 12 sequences, 110 cells | True |
| 1 | stationary_prior | large | -0.0084867 | -0.0189726 | 0.0019992 | 12 sequences, 32 cells | False |
| 1 | sgqf_gaussian | all | -0.0020667 | -0.0033647 | -0.0007687 | 12 sequences, 240 cells | True |
| 1 | sgqf_gaussian | near_zero | -0.0030270 | -0.0053124 | -0.0007415 | 12 sequences, 98 cells | True |
| 1 | sgqf_gaussian | ordinary | -0.0013672 | -0.0020904 | -0.0006439 | 12 sequences, 110 cells | True |
| 1 | sgqf_gaussian | large | -0.0016901 | -0.0037875 | 0.0004074 | 12 sequences, 32 cells | True |
| 1 | sgqf_joint | all | -0.0001780 | -0.0005838 | 0.0002277 | 12 sequences, 240 cells | True |
| 1 | sgqf_joint | near_zero | -0.0005013 | -0.0011010 | 0.0000983 | 12 sequences, 98 cells | True |
| 1 | sgqf_joint | ordinary | -0.0000319 | -0.0004294 | 0.0003656 | 12 sequences, 110 cells | True |
| 1 | sgqf_joint | large | 0.0002787 | -0.0005037 | 0.0010612 | 12 sequences, 32 cells | True |
| 4 | transition | all | -0.0042763 | N/A | N/A | 11 sequences, 880 cells | False |
| 4 | transition | near_zero | -0.0064136 | N/A | N/A | 11 sequences, 364 cells | False |
| 4 | transition | ordinary | -0.0022357 | N/A | N/A | 11 sequences, 397 cells | False |
| 4 | transition | large | -0.0046546 | N/A | N/A | 11 sequences, 119 cells | False |
| 4 | stationary_prior | all | -0.0189073 | N/A | N/A | 11 sequences, 880 cells | False |
| 4 | stationary_prior | near_zero | -0.0250530 | N/A | N/A | 11 sequences, 364 cells | False |
| 4 | stationary_prior | ordinary | -0.0124318 | N/A | N/A | 11 sequences, 397 cells | False |
| 4 | stationary_prior | large | -0.0202970 | N/A | N/A | 11 sequences, 119 cells | False |
| 4 | sgqf_gaussian | all | -0.0036801 | N/A | N/A | 11 sequences, 880 cells | False |
| 4 | sgqf_gaussian | near_zero | -0.0057259 | N/A | N/A | 11 sequences, 364 cells | False |
| 4 | sgqf_gaussian | ordinary | -0.0022194 | N/A | N/A | 11 sequences, 397 cells | False |
| 4 | sgqf_gaussian | large | -0.0020338 | N/A | N/A | 11 sequences, 119 cells | False |
| 4 | sgqf_joint | all | -0.0006089 | N/A | N/A | 11 sequences, 880 cells | False |
| 4 | sgqf_joint | near_zero | -0.0004766 | N/A | N/A | 11 sequences, 364 cells | False |
| 4 | sgqf_joint | ordinary | -0.0002790 | N/A | N/A | 11 sequences, 397 cells | False |
| 4 | sgqf_joint | large | -0.0014166 | N/A | N/A | 11 sequences, 119 cells | False |
