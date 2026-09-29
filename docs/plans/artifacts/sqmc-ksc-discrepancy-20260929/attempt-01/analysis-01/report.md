# KSC discrepancy diagnostic tables

UNTUNED FP64 GPU/XLA KSC discrepancy diagnostic; TF32 off; Contract E and dual-cap enabled. UNTUNED diagnostic perturbations. No production/HMC/default or method-ranking inference.

All intervals below are exploratory conditional intervals over eight designs on a fixed selected dataset. Retrospective dataset selection prevents population inference.

## Reference values and scores

| Dataset | T | log likelihood | gamma_raw score | log_beta score |
|---|---:|---:|---:|---:|
| 213001 | 10 | -22.855371012 | -0.364011242 | -0.422570929 |
| 213001 | 50 | -113.832453483 | -0.156559526 | -1.338362901 |
| 213001 | 120 | -278.788317999 | -0.406798194 | -1.602529336 |
| 213006 | 10 | -25.244929110 | 0.083074551 | -1.159335503 |
| 213006 | 50 | -112.120239487 | -0.815108535 | -1.460711504 |
| 213006 | 120 | -282.122686066 | 5.912217082 | -3.457001562 |

## Derivative checks

| Dataset | Route | T | Coordinate | Analytical | Status | Error on accepted FD steps |
|---|---|---:|---:|---:|---|---:|
| 213001 | IID | 10 | 0 | -0.7259615 | pass | 4.600764214046649e-11 |
| 213001 | IID | 10 | 1 | -0.23512 | pass | 5.204136288572414e-10 |
| 213001 | IID | 50 | 0 | -0.4247671 | pass | 5.26868326744534e-10 |
| 213001 | IID | 50 | 1 | -1.076673 | pass | 1.4632000056025163e-09 |
| 213001 | IID | 120 | 0 | -2.051522 | pass | 5.889906340428297e-09 |
| 213001 | IID | 120 | 1 | -1.483214 | pass | 2.9103315313250278e-09 |
| 213006 | IID | 10 | 0 | 0.01944004 | pass | 7.809390911717173e-10 |
| 213006 | IID | 10 | 1 | -1.128348 | pass | 4.050049184911586e-10 |
| 213006 | IID | 50 | 0 | -1.11472 | pass | 2.515550878712247e-09 |
| 213006 | IID | 50 | 1 | -1.226809 | pass | 3.430108863611281e-09 |
| 213006 | IID | 120 | 0 | 6.58532 | pass | 3.6829854721531774e-09 |
| 213006 | IID | 120 | 1 | -3.336833 | pass | 1.4050411056842904e-08 |
| 213001 | Inverse CDF | 10 | 0 | -0.4871887 | pass | 1.1499861063413164e-09 |
| 213001 | Inverse CDF | 10 | 1 | -0.2806874 | pass | 3.2841266928151924e-09 |
| 213001 | Inverse CDF | 50 | 0 | -0.03089843 | pass | 5.456423839839131e-08 |
| 213001 | Inverse CDF | 50 | 1 | -1.191188 | pass | 1.4815134763423998e-07 |
| 213001 | Inverse CDF | 120 | 0 | -0.266832 | pass | 7.498300310104788e-07 |
| 213001 | Inverse CDF | 120 | 1 | -1.677681 | pass | 4.830423629975655e-07 |
| 213006 | Inverse CDF | 10 | 0 | -0.020979 | pass | 9.808665824007079e-08 |
| 213006 | Inverse CDF | 10 | 1 | -1.040349 | pass | 3.296163852972711e-08 |
| 213006 | Inverse CDF | 50 | 0 | -0.4777389 | pass | 1.476716402493139e-07 |
| 213006 | Inverse CDF | 50 | 1 | -1.276265 | pass | 6.692159670862452e-07 |
| 213006 | Inverse CDF | 120 | 0 | 9.772707 | pass | 1.2102969986216294e-06 |
| 213006 | Inverse CDF | 120 | 1 | -4.098036 | pass | 1.1748069402628403e-05 |
| 213001 | Permutation .98 | 10 | 0 | -0.487191 | pass | 1.145402439561849e-09 |
| 213001 | Permutation .98 | 10 | 1 | -0.2812929 | pass | 4.626069693980384e-09 |
| 213001 | Permutation .98 | 50 | 0 | -0.03141541 | pass | 5.281084576452821e-07 |
| 213001 | Permutation .98 | 50 | 1 | -1.192035 | pass | 1.4406766357666356e-07 |
| 213001 | Permutation .98 | 120 | 0 | -0.2731993 | pass | 7.250856327856159e-06 |
| 213001 | Permutation .98 | 120 | 1 | -1.680463 | pass | 5.621254173249213e-08 |
| 213006 | Permutation .98 | 10 | 0 | -0.02059505 | pass | 4.0791926836369896e-08 |
| 213006 | Permutation .98 | 10 | 1 | -1.042619 | pass | 8.380628635507037e-09 |
| 213006 | Permutation .98 | 50 | 0 | -0.4886772 | pass | 6.524248086781093e-07 |
| 213006 | Permutation .98 | 50 | 1 | -1.279522 | pass | 7.046938721444462e-08 |
| 213006 | Permutation .98 | 120 | 0 | 9.699851 | pass | 1.0165744722456793e-05 |
| 213006 | Permutation .98 | 120 | 1 | -4.0923 | pass | 3.784288509223188e-07 |
| 213001 | Permutation .97 | 10 | 0 | -0.4865234 | pass | 1.975414998245384e-09 |
| 213001 | Permutation .97 | 10 | 1 | -0.2799759 | pass | 7.998899875794052e-10 |
| 213001 | Permutation .97 | 50 | 0 | -0.03182657 | pass | 7.997191672076687e-08 |
| 213001 | Permutation .97 | 50 | 1 | -1.190606 | pass | 1.7835605570226676e-07 |
| 213001 | Permutation .97 | 120 | 0 | -0.2673654 | pass | 2.3490638656475937e-06 |
| 213001 | Permutation .97 | 120 | 1 | -1.67901 | pass | 3.6283825055605234e-06 |
| 213006 | Permutation .97 | 10 | 0 | -0.02413608 | pass | 4.4135117782673206e-10 |
| 213006 | Permutation .97 | 10 | 1 | -1.039737 | pass | 6.037398581071329e-09 |
| 213006 | Permutation .97 | 50 | 0 | -0.4979128 | pass | 1.1896968760138904e-07 |
| 213006 | Permutation .97 | 50 | 1 | -1.277143 | pass | 4.9404309088174614e-08 |
| 213006 | Permutation .97 | 120 | 0 | 9.713087 | pass | 1.4626407143225606e-06 |
| 213006 | Permutation .97 | 120 | 1 | -4.097575 | pass | 3.2093831237034465e-07 |

## Conditional particle replication

| Dataset | Route | N | Mean logL error | Mean gamma error ± SE | Mean beta error ± SE | Mean norm error ± SE | Norm of mean error |
|---|---|---:|---:|---:|---:|---:|---:|
| 213001 | IID | 1008 | 0.0462545 | 0.315091 ± 0.261 | -0.121255 ± 0.0686 | 0.546677 ± 0.216 | 0.337617 |
| 213001 | IID | 2016 | 0.0930433 | 0.247368 ± 0.2 | -0.143042 ± 0.0499 | 0.514199 ± 0.128 | 0.285748 |
| 213001 | IID | 4032 | 0.0763191 | 0.173523 ± 0.117 | -0.133309 ± 0.0278 | 0.349318 ± 0.0625 | 0.218818 |
| 213001 | Inverse CDF | 1008 | 0.0455436 | 0.238756 ± 0.157 | -0.146843 ± 0.0296 | 0.429523 ± 0.102 | 0.280298 |
| 213001 | Inverse CDF | 2016 | 0.0129262 | 0.136025 ± 0.102 | -0.12249 ± 0.0192 | 0.282737 ± 0.0642 | 0.183048 |
| 213001 | Inverse CDF | 4032 | 0.111208 | 0.253173 ± 0.0672 | -0.181112 ± 0.0182 | 0.335483 ± 0.0511 | 0.311285 |
| 213001 | Permutation .98 | 1008 | 0.0479382 | 0.229494 ± 0.159 | -0.143244 ± 0.0297 | 0.429946 ± 0.101 | 0.27053 |
| 213001 | Permutation .98 | 2016 | 0.0140409 | 0.129272 ± 0.102 | -0.121225 ± 0.0204 | 0.27957 ± 0.064 | 0.17722 |
| 213001 | Permutation .98 | 4032 | 0.111397 | 0.254407 ± 0.0654 | -0.182882 ± 0.0183 | 0.33598 ± 0.0501 | 0.313319 |
| 213001 | Permutation .97 | 1008 | 0.0482916 | 0.226649 ± 0.159 | -0.144365 ± 0.0303 | 0.429512 ± 0.101 | 0.268721 |
| 213001 | Permutation .97 | 2016 | 0.0141099 | 0.133887 ± 0.099 | -0.119725 ± 0.0208 | 0.276258 ± 0.0627 | 0.179611 |
| 213001 | Permutation .97 | 4032 | 0.111772 | 0.257019 ± 0.0669 | -0.180966 ± 0.0179 | 0.337003 ± 0.0518 | 0.314337 |
| 213006 | IID | 1008 | -0.0441616 | 1.90209 ± 0.336 | -0.0820496 ± 0.0691 | 1.93779 ± 0.315 | 1.90386 |
| 213006 | IID | 2016 | 0.0908509 | 1.69751 ± 0.257 | -0.0506788 ± 0.0514 | 1.71043 ± 0.251 | 1.69827 |
| 213006 | IID | 4032 | 0.0104711 | 1.5607 ± 0.161 | -0.0427304 ± 0.0341 | 1.56376 ± 0.161 | 1.56129 |
| 213006 | Inverse CDF | 1008 | 0.41798 | 1.8257 ± 0.109 | -0.0721311 ± 0.0274 | 1.82834 ± 0.109 | 1.82712 |
| 213006 | Inverse CDF | 2016 | 0.289863 | 1.87651 ± 0.111 | -0.0725707 ± 0.0264 | 1.87883 ± 0.112 | 1.87791 |
| 213006 | Inverse CDF | 4032 | 0.242578 | 1.82235 ± 0.138 | -0.0659359 ± 0.0309 | 1.82484 ± 0.139 | 1.82354 |
| 213006 | Permutation .98 | 1008 | 0.417266 | 1.83019 ± 0.107 | -0.0755089 ± 0.0271 | 1.83293 ± 0.108 | 1.83175 |
| 213006 | Permutation .98 | 2016 | 0.289449 | 1.87768 ± 0.107 | -0.0742596 ± 0.0248 | 1.87995 ± 0.108 | 1.87915 |
| 213006 | Permutation .98 | 4032 | 0.242443 | 1.83831 ± 0.138 | -0.0676341 ± 0.0295 | 1.84072 ± 0.139 | 1.83956 |
| 213006 | Permutation .97 | 1008 | 0.416888 | 1.84392 ± 0.104 | -0.0745467 ± 0.0274 | 1.84662 ± 0.105 | 1.84543 |
| 213006 | Permutation .97 | 2016 | 0.288624 | 1.89081 ± 0.106 | -0.0740964 ± 0.0259 | 1.89315 ± 0.107 | 1.89226 |
| 213006 | Permutation .97 | 4032 | 0.24131 | 1.83947 ± 0.138 | -0.0701387 ± 0.0297 | 1.84196 ± 0.139 | 1.84081 |

## Numerical-control sensitivity

| Dataset | Route | Arm | Mean logL error | Mean norm score error |
|---|---|---|---:|---:|
| 213001 | IID | baseline | 0.304936 | 0.329565 |
| 213001 | IID | flow_16 | 0.295171 | 0.324321 |
| 213001 | IID | flow_32 | 0.289962 | 0.321459 |
| 213001 | IID | flow_64 | 0.287277 | 0.31977 |
| 213001 | IID | transport_96_48 | 0.304936 | 0.329565 |
| 213001 | IID | transport_384_192 | 0.304936 | 0.329565 |
| 213001 | IID | epsilon_25.6 | 0.352014 | 0.317676 |
| 213001 | IID | epsilon_409.6 | 0.297262 | 0.3237 |
| 213001 | IID | combined_refined | 0.287277 | 0.31977 |
| 213001 | Inverse CDF | baseline | 0.448461 | 0.44972 |
| 213001 | Inverse CDF | flow_16 | 0.451278 | 0.458594 |
| 213001 | Inverse CDF | flow_32 | 0.452273 | 0.469012 |
| 213001 | Inverse CDF | flow_64 | 0.452455 | 0.510994 |
| 213001 | Inverse CDF | transport_96_48 | 0.448461 | 0.44972 |
| 213001 | Inverse CDF | transport_384_192 | 0.448462 | 0.449718 |
| 213001 | Inverse CDF | epsilon_25.6 | 0.438405 | 0.509028 |
| 213001 | Inverse CDF | epsilon_409.6 | 0.464892 | 0.455101 |
| 213001 | Inverse CDF | combined_refined | 0.452455 | 0.510994 |
| 213001 | Permutation .98 | baseline | 0.447735 | 0.446382 |
| 213001 | Permutation .98 | flow_16 | 0.449901 | 0.472237 |
| 213001 | Permutation .98 | flow_32 | 0.450717 | 0.504044 |
| 213001 | Permutation .98 | flow_64 | 0.451091 | 0.521816 |
| 213001 | Permutation .98 | transport_96_48 | 0.447735 | 0.446382 |
| 213001 | Permutation .98 | transport_384_192 | 0.447735 | 0.446382 |
| 213001 | Permutation .98 | epsilon_25.6 | 0.438221 | 0.484251 |
| 213001 | Permutation .98 | epsilon_409.6 | 0.464289 | 0.433822 |
| 213001 | Permutation .98 | combined_refined | 0.451091 | 0.521816 |
| 213001 | Permutation .97 | baseline | 0.448224 | 0.439782 |
| 213001 | Permutation .97 | flow_16 | 0.450621 | 0.486546 |
| 213001 | Permutation .97 | flow_32 | 0.451523 | 0.49571 |
| 213001 | Permutation .97 | flow_64 | 0.45186 | 0.503763 |
| 213001 | Permutation .97 | transport_96_48 | 0.448224 | 0.439782 |
| 213001 | Permutation .97 | transport_384_192 | 0.448224 | 0.439782 |
| 213001 | Permutation .97 | epsilon_25.6 | 0.43871 | 0.492016 |
| 213001 | Permutation .97 | epsilon_409.6 | 0.465496 | 0.459062 |
| 213001 | Permutation .97 | combined_refined | 0.45186 | 0.503763 |
| 213006 | IID | baseline | 0.0452193 | 2.07211 |
| 213006 | IID | flow_16 | 0.0490478 | 2.07535 |
| 213006 | IID | flow_32 | 0.0509864 | 2.07707 |
| 213006 | IID | flow_64 | 0.051954 | 2.0783 |
| 213006 | IID | transport_96_48 | 0.0452193 | 2.07211 |
| 213006 | IID | transport_384_192 | 0.0452193 | 2.07211 |
| 213006 | IID | epsilon_25.6 | -0.00416987 | 2.24053 |
| 213006 | IID | epsilon_409.6 | 0.0887397 | 2.05694 |
| 213006 | IID | combined_refined | 0.051954 | 2.0783 |
| 213006 | Inverse CDF | baseline | 0.396051 | 1.85387 |
| 213006 | Inverse CDF | flow_16 | 0.414766 | 1.88061 |
| 213006 | Inverse CDF | flow_32 | 0.424984 | 1.85021 |
| 213006 | Inverse CDF | flow_64 | 0.430159 | 1.84587 |
| 213006 | Inverse CDF | transport_96_48 | 0.396051 | 1.85387 |
| 213006 | Inverse CDF | transport_384_192 | 0.396051 | 1.85387 |
| 213006 | Inverse CDF | epsilon_25.6 | 0.329904 | 1.94323 |
| 213006 | Inverse CDF | epsilon_409.6 | 0.448355 | 1.87393 |
| 213006 | Inverse CDF | combined_refined | 0.430159 | 1.84587 |
| 213006 | Permutation .98 | baseline | 0.393742 | 1.8647 |
| 213006 | Permutation .98 | flow_16 | 0.411815 | 1.91617 |
| 213006 | Permutation .98 | flow_32 | 0.421123 | 1.90081 |
| 213006 | Permutation .98 | flow_64 | 0.426121 | 1.89521 |
| 213006 | Permutation .98 | transport_96_48 | 0.393742 | 1.8647 |
| 213006 | Permutation .98 | transport_384_192 | 0.393742 | 1.8647 |
| 213006 | Permutation .98 | epsilon_25.6 | 0.326268 | 1.93471 |
| 213006 | Permutation .98 | epsilon_409.6 | 0.447556 | 1.9027 |
| 213006 | Permutation .98 | combined_refined | 0.426121 | 1.89521 |
| 213006 | Permutation .97 | baseline | 0.392105 | 1.87708 |
| 213006 | Permutation .97 | flow_16 | 0.409825 | 1.91244 |
| 213006 | Permutation .97 | flow_32 | 0.419125 | 1.91394 |
| 213006 | Permutation .97 | flow_64 | 0.424066 | 1.87446 |
| 213006 | Permutation .97 | transport_96_48 | 0.392105 | 1.87708 |
| 213006 | Permutation .97 | transport_384_192 | 0.392105 | 1.87708 |
| 213006 | Permutation .97 | epsilon_25.6 | 0.324735 | 1.93575 |
| 213006 | Permutation .97 | epsilon_409.6 | 0.444307 | 1.92391 |
| 213006 | Permutation .97 | combined_refined | 0.424066 | 1.87446 |

## Decision and inference status

| Item | Status |
|---|---|
| Hard validity vetoes | 0 invalid evaluated particle cells |
| Derivative classification | {'pass': 48} |
| Statistically supported method ranking | Not tested; all four retained |
| Descriptive differences | Numerical control arms have two designs; report individually |
| Default readiness | Not established |
| Next evidence | Causal localization depends on the recorded FD and convergence findings |

The reader-facing result note supplies causal interpretation and terminal review. Full numerical rows, signed coordinate errors, SD/SE and exploratory intervals remain in the JSON/CSV files.
