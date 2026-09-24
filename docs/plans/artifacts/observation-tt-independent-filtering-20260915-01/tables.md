# A06 diagnostic tables

Primary values are equal-sequence averages of normalized MSE, not pooled time-point averages. Runtime and evidence diagnostics are descriptive.
All methods use the same sequence set within a dimension: all 12 d1 sequences; 11 d4 sequences excluding failed guide sequence 8. The d4 table is conditional on completion and cannot support population ranking. All available outcomes, including the two unguided methods on sequence 8, remain in report.json.

| d | method | all | near zero | ordinary | large | median fit s | median total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | transition | 0.0021600 | 0.0030579 | 0.0011964 | 0.0020561 | 0.001 | 0.434 |
| 1 | stationary_prior | 0.0051131 | 0.0056793 | 0.0018754 | 0.0098854 | 0.000 | 0.547 |
| 1 | sgqf_gaussian | 0.0037369 | 0.0049775 | 0.0027870 | 0.0030887 | 0.000 | 0.462 |
| 1 | sgqf_joint | 0.0018482 | 0.0024519 | 0.0014518 | 0.0011200 | 0.170 | 0.647 |
| 1 | tt_predictive | 0.0015124 | 0.0019170 | 0.0012909 | 0.0007813 | 3.732 | 4.650 |
| 1 | tt_guided | 0.0016963 | 0.0018042 | 0.0013399 | 0.0017197 | 3.716 | 4.594 |
| 1 | tt_pair_block | 0.0016075 | 0.0017946 | 0.0012691 | 0.0015217 | 2.462 | 3.567 |
| 1 | tt_sgqf_safeguard | 0.0016702 | 0.0019506 | 0.0014199 | 0.0013987 | 5.594 | 6.593 |
| 4 | transition | 0.0066377 | 0.0093457 | 0.0041904 | 0.0065428 | 0.001 | 0.426 |
| 4 | stationary_prior | 0.0212687 | 0.0279851 | 0.0143865 | 0.0221852 | 0.000 | 0.567 |
| 4 | sgqf_gaussian | 0.0060415 | 0.0086580 | 0.0041741 | 0.0039220 | 0.000 | 0.483 |
| 4 | sgqf_joint | 0.0029703 | 0.0034087 | 0.0022337 | 0.0033048 | 0.189 | 0.665 |
| 4 | tt_predictive | 0.0072739 | 0.0105720 | 0.0052669 | 0.0038474 | 15.020 | 16.651 |
| 4 | tt_guided | 0.0066699 | 0.0090306 | 0.0053657 | 0.0034635 | 14.354 | 15.869 |
| 4 | tt_pair_block | 0.0026151 | 0.0032617 | 0.0021797 | 0.0020670 | 24.811 | 27.088 |
| 4 | tt_sgqf_safeguard | 0.0023614 | 0.0029321 | 0.0019547 | 0.0018882 | 43.520 | 45.624 |

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
