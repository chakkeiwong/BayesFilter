# full_d3_T120: actual Kalman and particle scores

Program: FP64 GPU/XLA reference comparison, TF32 disabled. Each route calibrates flow substeps 2 versus 8 within this exact scope. Full-model repair scopes also calibrate terminal balancing; the other numerical protections remain inherited hypotheses. Tuning paths are in audit.json. This is not FP32/TF32 production evidence or a statistically supported ranking. Reused pilot observations are disclosed in the run manifest. Invalid rows are retained for diagnosis and excluded from error summaries.

N=1020, d=3, T=120. Q scores use lower-Cholesky coordinates, with log diagonal; they are not derivatives with respect to covariance entries.

## Data 201001, filter 202001

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

Kalman log likelihood: -333.9249253.

| Route | Log likelihood | Absolute error | Valid |
|---|---:|---:|---|
| IID | -334.4008293 | 0.4759040052 | True |
| Inverse CDF | not evaluated | unavailable | no |
| Permutation | not evaluated | unavailable | no |
| Permutation cap .97 | not evaluated | unavailable | no |

| Coordinate | Kalman | IID | Inverse CDF | Permutation | Cap .97 |
|---|---:|---:|---:|---:|---:|
| A[1,1] | 1.028538852 | 0.6789688971 (0.3495699549) | not evaluated | not evaluated | not evaluated |
| A[1,2] | 10.41154861 | 9.776788114 (0.6347604973) | not evaluated | not evaluated | not evaluated |
| A[1,3] | 3.310904746 | 4.038699813 (0.7277950676) | not evaluated | not evaluated | not evaluated |
| A[2,1] | 6.234063841 | 6.262226834 (0.02816299386) | not evaluated | not evaluated | not evaluated |
| A[2,2] | -14.4219232 | -15.1861191 (0.7641958988) | not evaluated | not evaluated | not evaluated |
| A[2,3] | 12.19661669 | 12.93875126 (0.742134568) | not evaluated | not evaluated | not evaluated |
| A[3,1] | -7.978405919 | -7.884429221 (0.09397669748) | not evaluated | not evaluated | not evaluated |
| A[3,2] | 5.548737522 | 6.088722806 (0.5399852833) | not evaluated | not evaluated | not evaluated |
| A[3,3] | 2.154232164 | 1.954938907 (0.199293257) | not evaluated | not evaluated | not evaluated |
| log_LQ[1,1] | 4.800293802 | 5.223122001 (0.4228281993) | not evaluated | not evaluated | not evaluated |
| LQ[2,1] | -2.7376979 | -3.024350201 (0.2866523011) | not evaluated | not evaluated | not evaluated |
| log_LQ[2,2] | 2.639054365 | 3.308126719 (0.6690723538) | not evaluated | not evaluated | not evaluated |
| LQ[3,1] | -19.95825828 | -20.4636275 (0.505369218) | not evaluated | not evaluated | not evaluated |
| LQ[3,2] | -27.76339199 | -28.70752614 (0.9441341526) | not evaluated | not evaluated | not evaluated |
| log_LQ[3,3] | 6.361850039 | 6.451613812 (0.08976377303) | not evaluated | not evaluated | not evaluated |
| log_R_variance | 6.629710436 | 6.595950823 (0.03375961311) | not evaluated | not evaluated | not evaluated |
| initial_mean_scale | 2.08709875 | 2.121344133 (0.03424538236) | not evaluated | not evaluated | not evaluated |

## Data 201001, filter 202002

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

Kalman log likelihood: -333.9249253.

| Route | Log likelihood | Absolute error | Valid |
|---|---:|---:|---|
| IID | -334.2299222 | 0.3049968508 | True |
| Inverse CDF | not evaluated | unavailable | no |
| Permutation | not evaluated | unavailable | no |
| Permutation cap .97 | not evaluated | unavailable | no |

| Coordinate | Kalman | IID | Inverse CDF | Permutation | Cap .97 |
|---|---:|---:|---:|---:|---:|
| A[1,1] | 1.028538852 | 0.3996370924 (0.6289017596) | not evaluated | not evaluated | not evaluated |
| A[1,2] | 10.41154861 | 11.17579867 (0.7642500581) | not evaluated | not evaluated | not evaluated |
| A[1,3] | 3.310904746 | 2.70615012 (0.6047546259) | not evaluated | not evaluated | not evaluated |
| A[2,1] | 6.234063841 | 6.486196613 (0.2521327722) | not evaluated | not evaluated | not evaluated |
| A[2,2] | -14.4219232 | -15.20339599 (0.7814727857) | not evaluated | not evaluated | not evaluated |
| A[2,3] | 12.19661669 | 12.43604494 (0.2394282424) | not evaluated | not evaluated | not evaluated |
| A[3,1] | -7.978405919 | -8.59621993 (0.6178140113) | not evaluated | not evaluated | not evaluated |
| A[3,2] | 5.548737522 | 5.741329052 (0.1925915295) | not evaluated | not evaluated | not evaluated |
| A[3,3] | 2.154232164 | 1.993806344 (0.1604258195) | not evaluated | not evaluated | not evaluated |
| log_LQ[1,1] | 4.800293802 | 5.425174926 (0.6248811242) | not evaluated | not evaluated | not evaluated |
| LQ[2,1] | -2.7376979 | -3.380510083 (0.6428121831) | not evaluated | not evaluated | not evaluated |
| log_LQ[2,2] | 2.639054365 | 3.165235412 (0.5261810468) | not evaluated | not evaluated | not evaluated |
| LQ[3,1] | -19.95825828 | -19.23891158 (0.7193466993) | not evaluated | not evaluated | not evaluated |
| LQ[3,2] | -27.76339199 | -27.98107 (0.2176780136) | not evaluated | not evaluated | not evaluated |
| log_LQ[3,3] | 6.361850039 | 6.498463438 (0.1366133992) | not evaluated | not evaluated | not evaluated |
| log_R_variance | 6.629710436 | 6.552083802 (0.07762663357) | not evaluated | not evaluated | not evaluated |
| initial_mean_scale | 2.08709875 | 2.269376177 (0.1822774269) | not evaluated | not evaluated | not evaluated |

## Data 201002, filter 202001

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

Kalman log likelihood: -317.670838.

| Route | Log likelihood | Absolute error | Valid |
|---|---:|---:|---|
| IID | -317.7630831 | 0.09224507289 | True |
| Inverse CDF | not evaluated | unavailable | no |
| Permutation | not evaluated | unavailable | no |
| Permutation cap .97 | not evaluated | unavailable | no |

| Coordinate | Kalman | IID | Inverse CDF | Permutation | Cap .97 |
|---|---:|---:|---:|---:|---:|
| A[1,1] | -19.84892221 | -19.81231219 (0.03661002963) | not evaluated | not evaluated | not evaluated |
| A[1,2] | 10.67611871 | 10.69151244 (0.01539372662) | not evaluated | not evaluated | not evaluated |
| A[1,3] | -4.387731991 | -4.837525444 (0.4497934534) | not evaluated | not evaluated | not evaluated |
| A[2,1] | -16.18946333 | -15.73530392 (0.4541594099) | not evaluated | not evaluated | not evaluated |
| A[2,2] | 0.5374266458 | 0.7165434794 (0.1791168336) | not evaluated | not evaluated | not evaluated |
| A[2,3] | 2.579788647 | 2.219575733 (0.3602129143) | not evaluated | not evaluated | not evaluated |
| A[3,1] | -14.06602082 | -14.08330867 (0.01728785651) | not evaluated | not evaluated | not evaluated |
| A[3,2] | -0.2524526744 | -0.4450739163 (0.1926212419) | not evaluated | not evaluated | not evaluated |
| A[3,3] | -14.49344293 | -14.89739194 (0.403949011) | not evaluated | not evaluated | not evaluated |
| log_LQ[1,1] | -7.715542061 | -7.724560048 (0.009017986558) | not evaluated | not evaluated | not evaluated |
| LQ[2,1] | 13.42185426 | 13.65249308 (0.2306388226) | not evaluated | not evaluated | not evaluated |
| log_LQ[2,2] | 5.772913428 | 5.789460154 (0.01654672628) | not evaluated | not evaluated | not evaluated |
| LQ[3,1] | -1.644577251 | -1.325254227 (0.3193230246) | not evaluated | not evaluated | not evaluated |
| LQ[3,2] | 5.880238486 | 6.274892573 (0.3946540875) | not evaluated | not evaluated | not evaluated |
| log_LQ[3,3] | -6.210077839 | -6.000921045 (0.2091567939) | not evaluated | not evaluated | not evaluated |
| log_R_variance | 3.110898792 | 3.138952872 (0.02805407994) | not evaluated | not evaluated | not evaluated |
| initial_mean_scale | 0.7170785472 | 0.7300824181 (0.01300387087) | not evaluated | not evaluated | not evaluated |

## Data 201002, filter 202002

Each method entry is its actual score followed by absolute error in parentheses. CSV files preserve full floating-point precision.

Kalman log likelihood: -317.670838.

| Route | Log likelihood | Absolute error | Valid |
|---|---:|---:|---|
| IID | -317.6034267 | 0.0674112947 | True |
| Inverse CDF | not evaluated | unavailable | no |
| Permutation | not evaluated | unavailable | no |
| Permutation cap .97 | not evaluated | unavailable | no |

| Coordinate | Kalman | IID | Inverse CDF | Permutation | Cap .97 |
|---|---:|---:|---:|---:|---:|
| A[1,1] | -19.84892221 | -20.3538085 (0.5048862891) | not evaluated | not evaluated | not evaluated |
| A[1,2] | 10.67611871 | 10.72777148 (0.05165277162) | not evaluated | not evaluated | not evaluated |
| A[1,3] | -4.387731991 | -4.317388335 (0.0703436559) | not evaluated | not evaluated | not evaluated |
| A[2,1] | -16.18946333 | -16.73545087 (0.5459875436) | not evaluated | not evaluated | not evaluated |
| A[2,2] | 0.5374266458 | 0.9756649793 (0.4382383335) | not evaluated | not evaluated | not evaluated |
| A[2,3] | 2.579788647 | 2.866222383 (0.2864337361) | not evaluated | not evaluated | not evaluated |
| A[3,1] | -14.06602082 | -13.94449934 (0.1215214792) | not evaluated | not evaluated | not evaluated |
| A[3,2] | -0.2524526744 | -0.1849128197 (0.06753985474) | not evaluated | not evaluated | not evaluated |
| A[3,3] | -14.49344293 | -15.03258042 (0.5391374819) | not evaluated | not evaluated | not evaluated |
| log_LQ[1,1] | -7.715542061 | -7.728397815 (0.0128557541) | not evaluated | not evaluated | not evaluated |
| LQ[2,1] | 13.42185426 | 13.82124757 (0.3993933094) | not evaluated | not evaluated | not evaluated |
| log_LQ[2,2] | 5.772913428 | 5.621140552 (0.1517728755) | not evaluated | not evaluated | not evaluated |
| LQ[3,1] | -1.644577251 | -1.541070343 (0.1035069083) | not evaluated | not evaluated | not evaluated |
| LQ[3,2] | 5.880238486 | 5.752559648 (0.1276788377) | not evaluated | not evaluated | not evaluated |
| log_LQ[3,3] | -6.210077839 | -5.911511388 (0.2985664509) | not evaluated | not evaluated | not evaluated |
| log_R_variance | 3.110898792 | 2.966726634 (0.1441721575) | not evaluated | not evaluated | not evaluated |
| initial_mean_scale | 0.7170785472 | 0.6929675738 (0.02411097338) | not evaluated | not evaluated | not evaluated |

