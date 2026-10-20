# Corrected KSC comparison: full seven-component mixture

Configuration: FP64 GPU/XLA diagnostic variants, TF32 off. Particle N=1008; each route/horizon uses its original exact-scope frozen tuning (paths in every cell and particle-scopes.json). The deterministic Gaussian-sum Kalman reference has all seven observation components and a checked quadrature resolution. This is not production FP32/TF32, native SV, HMC validation or a default change.

The single-Gaussian column in the earlier report computes a different likelihood and is removed from this main comparison. The new Gaussian-sum reference uses the same full mixture as the particle methods. All original particle evaluations are preserved.

## Actual likelihoods and scores

Means over the same eight datasets. Errors are measured per dataset before averaging. Reference self-errors are left blank rather than presented as exact zeros. Both parameter coordinates are retained.

| T | Method | Mean log likelihood | Mean gamma_raw score | Mean log_beta score | Mean absolute log-likelihood error | Mean absolute gamma error | Mean absolute log_beta error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | Full-mixture Gaussian-sum Kalman reference | -23.694267 | -0.080417191 | 0.1407618 | reference | reference | reference |
| 10 | IID | -23.767888 | -0.14053641 | 0.22713101 | 0.11646868 | 0.11098988 | 0.14952729 |
| 10 | Inverse CDF | -23.738327 | -0.10595006 | 0.20128545 | 0.096542311 | 0.085220831 | 0.11238896 |
| 10 | Permutation .98 | -23.73773 | -0.10501644 | 0.20225259 | 0.09678064 | 0.08429546 | 0.11363101 |
| 10 | Permutation .97 | -23.738043 | -0.10609436 | 0.20274705 | 0.097286726 | 0.08592102 | 0.11429613 |
| 20 | Full-mixture Gaussian-sum Kalman reference | -46.576561 | 0.71034536 | -0.39021103 | reference | reference | reference |
| 20 | IID | -47.293211 | 0.70732055 | -0.50805342 | 0.72704436 | 0.18601567 | 0.26437805 |
| 20 | Inverse CDF | -46.778666 | 0.68356083 | -0.41536413 | 0.20916652 | 0.1739932 | 0.17501898 |
| 20 | Permutation .98 | -46.778742 | 0.68711736 | -0.41475314 | 0.20931168 | 0.17367147 | 0.17446813 |
| 20 | Permutation .97 | -46.779399 | 0.68319914 | -0.41635767 | 0.21007543 | 0.17103693 | 0.17531414 |
| 50 | Full-mixture Gaussian-sum Kalman reference | -117.0546 | -0.27051666 | 0.041234721 | reference | reference | reference |
| 50 | IID | -117.53825 | -0.4002906 | 0.11471576 | 0.55222175 | 0.40437198 | 0.17618054 |
| 50 | Inverse CDF | -117.21812 | -0.29745983 | 0.076052206 | 0.59969732 | 0.5076503 | 0.19382551 |
| 50 | Permutation .98 | -117.21865 | -0.29684082 | 0.077110443 | 0.5994738 | 0.51365104 | 0.19304241 |
| 50 | Permutation .97 | -117.21871 | -0.3026147 | 0.076295577 | 0.60067761 | 0.5103928 | 0.19354765 |
| 120 | Full-mixture Gaussian-sum Kalman reference | -281.08123 | 0.14771746 | -0.75825504 | reference | reference | reference |
| 120 | IID | -282.76975 | -0.87222868 | -0.52489326 | 1.688522 | 1.4662613 | 0.37002083 |
| 120 | Inverse CDF | -282.3671 | -0.085924223 | -0.70532314 | 1.2858675 | 1.5050246 | 0.43378797 |
| 120 | Permutation .98 | -282.36767 | -0.11141053 | -0.70626358 | 1.2864373 | 1.4695562 | 0.43578134 |
| 120 | Permutation .97 | -282.37153 | -0.10275563 | -0.70814821 | 1.2903005 | 1.4933101 | 0.43431496 |

## Score-vector error and uncertainty

e_i=||score_i-reference_i||_2. SD is the sample standard deviation of e_i; SE=SD/sqrt(8) estimates uncertainty in its mean across independent dataset/design pairs. This is not fixed-dataset Monte Carlo uncertainty or the norm of the mean error vector.

| T | Method | Mean L2 error | SD | SE | 95% t interval for mean |
| --- | --- | --- | --- | --- | --- |
| 10 | IID | 0.2133033 | 0.12494643 | 0.044175235 | [0.10884547, 0.31776113] |
| 10 | Inverse CDF | 0.16294955 | 0.12141817 | 0.042927805 | [0.061441419, 0.26445768] |
| 10 | Permutation .98 | 0.16359784 | 0.12093807 | 0.042758063 | [0.062491087, 0.26470459] |
| 10 | Permutation .97 | 0.16503924 | 0.12192009 | 0.043105261 | [0.063111498, 0.26696699] |
| 20 | IID | 0.35479195 | 0.24516385 | 0.086678511 | [0.14982984, 0.55975405] |
| 20 | Inverse CDF | 0.27693823 | 0.17471228 | 0.061770119 | [0.1308751, 0.42300135] |
| 20 | Permutation .98 | 0.27720453 | 0.17569939 | 0.062119115 | [0.13031616, 0.4240929] |
| 20 | Permutation .97 | 0.27644516 | 0.17840268 | 0.063074872 | [0.12729679, 0.42559353] |
| 50 | IID | 0.46267345 | 0.25447554 | 0.08997069 | [0.24992658, 0.67542033] |
| 50 | Inverse CDF | 0.56874548 | 0.35456364 | 0.12535718 | [0.27232286, 0.8651681] |
| 50 | Permutation .98 | 0.57420214 | 0.35411066 | 0.12519702 | [0.27815822, 0.87024606] |
| 50 | Permutation .97 | 0.5717152 | 0.35633939 | 0.125985 | [0.27380801, 0.86962239] |
| 120 | IID | 1.5436218 | 1.1502173 | 0.40666322 | [0.58201613, 2.5052276] |
| 120 | Inverse CDF | 1.6081597 | 1.243907 | 0.43978755 | [0.56822735, 2.648092] |
| 120 | Permutation .98 | 1.5889494 | 1.2290178 | 0.4345234 | [0.56146488, 2.616434] |
| 120 | Permutation .97 | 1.6014946 | 1.2261853 | 0.43352196 | [0.57637806, 2.6266111] |

## Independent verification of the full-mixture reference

The first update retains seven exact Gaussian branches. Later updates retain 7*M branches from M quadrature atoms, with all seven observation components intact. At M=1201 this is 8407 branches before projection. Fixed quadrature projection is an approximation; it was checked at M=401,801,1201 on [-40,40] and M=1201 on [-48,48]. The comparator is not an exact seven-component posterior filter.

| T | Maximum absolute likelihood discrepancy vs independent grid | Maximum score L2 discrepancy vs independent grid |
| --- | --- | --- |
| 10 | 1.0658141e-14 | 2.2232199e-15 |
| 20 | 2.1316282e-14 | 3.7120164e-15 |
| 50 | 5.6843419e-14 | 6.94777e-15 |
| 120 | 5.1159077e-13 | 9.8102513e-15 |

Maximum refinement differences: likelihood 5.68434e-14, score coordinate 2.17604e-14. These observed discrepancies are not rigorous error bounds.
Maximum change to a previously reported particle score-vector error: 6.66134e-15. Candidate accuracy findings remain unchanged at the reported precision.

## Paired differences

Left minus right score-vector error, paired by saved dataset/design seed. Negative favors left. Intervals are exploratory and unadjusted for multiple comparisons. Inclusion of zero does not prove equivalence.

| T | Left | Right | Mean difference | SE | 95% paired t interval |
| --- | --- | --- | --- | --- | --- |
| 10 | IID | Inverse CDF | 0.050353749 | 0.017205361 | [0.009669535, 0.091037963] |
| 10 | IID | Permutation .98 | 0.049705458 | 0.016457487 | [0.010789684, 0.088621232] |
| 10 | IID | Permutation .97 | 0.048264053 | 0.016823884 | [0.0084818903, 0.088046216] |
| 10 | Inverse CDF | Permutation .98 | -0.00064829088 | 0.0010911808 | [-0.0032285235, 0.0019319418] |
| 10 | Inverse CDF | Permutation .97 | -0.0020896959 | 0.0014630515 | [-0.005549263, 0.0013698711] |
| 10 | Permutation .98 | Permutation .97 | -0.0014414051 | 0.00089206866 | [-0.0035508123, 0.00066800213] |
| 20 | IID | Inverse CDF | 0.077853719 | 0.0702324 | [-0.088219517, 0.24392695] |
| 20 | IID | Permutation .98 | 0.077587414 | 0.06944857 | [-0.086632358, 0.24180719] |
| 20 | IID | Permutation .97 | 0.078346788 | 0.068558497 | [-0.083768296, 0.24046187] |
| 20 | Inverse CDF | Permutation .98 | -0.00026630461 | 0.002181524 | [-0.0054247891, 0.0048921799] |
| 20 | Inverse CDF | Permutation .97 | 0.0004930689 | 0.0035262789 | [-0.0078452556, 0.0088313934] |
| 20 | Permutation .98 | Permutation .97 | 0.00075937352 | 0.003961746 | [-0.0086086672, 0.010127414] |
| 50 | IID | Inverse CDF | -0.10607203 | 0.10387742 | [-0.3517031, 0.13955904] |
| 50 | IID | Permutation .98 | -0.11152869 | 0.10283621 | [-0.35469769, 0.13164031] |
| 50 | IID | Permutation .97 | -0.10904174 | 0.10335234 | [-0.35343118, 0.13534769] |
| 50 | Inverse CDF | Permutation .98 | -0.0054566578 | 0.0035555057 | [-0.013864093, 0.0029507773] |
| 50 | Inverse CDF | Permutation .97 | -0.0029697142 | 0.0033095604 | [-0.010795581, 0.0048561525] |
| 50 | Permutation .98 | Permutation .97 | 0.0024869435 | 0.00244831 | [-0.0033023898, 0.0082762768] |
| 120 | IID | Inverse CDF | -0.06453781 | 0.50569959 | [-1.2603273, 1.1312517] |
| 120 | IID | Permutation .98 | -0.0453276 | 0.49810798 | [-1.2231658, 1.1325106] |
| 120 | IID | Permutation .97 | -0.057872753 | 0.49990453 | [-1.2399591, 1.1242136] |
| 120 | Inverse CDF | Permutation .98 | 0.01921021 | 0.0093315277 | [-0.002855347, 0.041275767] |
| 120 | Inverse CDF | Permutation .97 | 0.0066650566 | 0.009086664 | [-0.01482149, 0.028151603] |
| 120 | Permutation .98 | Permutation .97 | -0.012545153 | 0.0067157311 | [-0.028425334, 0.0033350274] |

## Conditional heuristic checks

Zero score, first-observation-only score and IID are checked per dataset and horizon; observed losses are promotion vetoes in those situations, not evidence for a universal ranking. The old different-model single-Gaussian heuristic is preserved in the historical report rather than used as this comparison baseline.

| T | Method | Heuristic | Observed losses / 8 |
| --- | --- | --- | --- |
| 10 | IID | zero_score | 0 |
| 10 | IID | first_observation_only | 0 |
| 10 | Inverse CDF | zero_score | 0 |
| 10 | Inverse CDF | first_observation_only | 0 |
| 10 | Inverse CDF | iid_dual_cap | 1 |
| 10 | Permutation .98 | zero_score | 0 |
| 10 | Permutation .98 | first_observation_only | 0 |
| 10 | Permutation .98 | iid_dual_cap | 1 |
| 10 | Permutation .97 | zero_score | 0 |
| 10 | Permutation .97 | first_observation_only | 0 |
| 10 | Permutation .97 | iid_dual_cap | 1 |
| 20 | IID | zero_score | 0 |
| 20 | IID | first_observation_only | 1 |
| 20 | Inverse CDF | zero_score | 0 |
| 20 | Inverse CDF | first_observation_only | 1 |
| 20 | Inverse CDF | iid_dual_cap | 2 |
| 20 | Permutation .98 | zero_score | 0 |
| 20 | Permutation .98 | first_observation_only | 1 |
| 20 | Permutation .98 | iid_dual_cap | 2 |
| 20 | Permutation .97 | zero_score | 0 |
| 20 | Permutation .97 | first_observation_only | 1 |
| 20 | Permutation .97 | iid_dual_cap | 2 |
| 50 | IID | zero_score | 0 |
| 50 | IID | first_observation_only | 1 |
| 50 | Inverse CDF | zero_score | 0 |
| 50 | Inverse CDF | first_observation_only | 0 |
| 50 | Inverse CDF | iid_dual_cap | 6 |
| 50 | Permutation .98 | zero_score | 0 |
| 50 | Permutation .98 | first_observation_only | 0 |
| 50 | Permutation .98 | iid_dual_cap | 6 |
| 50 | Permutation .97 | zero_score | 0 |
| 50 | Permutation .97 | first_observation_only | 0 |
| 50 | Permutation .97 | iid_dual_cap | 6 |
| 120 | IID | zero_score | 1 |
| 120 | IID | first_observation_only | 2 |
| 120 | Inverse CDF | zero_score | 1 |
| 120 | Inverse CDF | first_observation_only | 1 |
| 120 | Inverse CDF | iid_dual_cap | 4 |
| 120 | Permutation .98 | zero_score | 1 |
| 120 | Permutation .98 | first_observation_only | 1 |
| 120 | Permutation .98 | iid_dual_cap | 4 |
| 120 | Permutation .97 | zero_score | 1 |
| 120 | Permutation .97 | first_observation_only | 1 |
| 120 | Permutation .97 | iid_dual_cap | 4 |

## Decision and terminal review

The full-mixture Gaussian-sum calculation agrees with the independent density-grid reference and passes exact short-sequence, finite-difference and graph/XLA checks. The original particle errors therefore remain valid. The correction replaces the misleading main comparator, not the particle data. The former Gaussian-versus-particle score observation remains only a descriptive different-model heuristic finding, not a verdict about full-KSC Kalman filtering.

All four particle methods remain research candidates. Only the T=10 IID-versus-SQMC exploratory paired intervals exclude zero; no interval orders the three SQMC variants. Large finite particle errors remain accuracy findings, not infrastructure failures. One regime, eight pairs, finite particle count and limited scope-specific control search do not establish general superiority or readiness.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept corrected reference | All 32 datasets converge and agree with independent implementation | No reference validity/derivative veto | Empirical quadrature checks, not rigorous bounds | Use corrected full-mixture tables | No exact finite-M or native-SV claim |
| Retain all four routes | 128 preserved valid evaluations with recomputed errors | Conditional heuristic vetoes remain | Eight pairs, one regime, finite controls/N | Fresh tuning/replication before selecting a method | No default or overall winner |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Reference and original particle validity pass; per-case heuristic losses in heuristics.csv prohibit promotion there |
| Statistically supported ranking | Only exploratory T10 SQMC-versus-IID intervals; no overall or within-SQMC ranking |
| Descriptive differences | Other means, likelihood errors, timing and tails |
| Default readiness | Not established |
| Next evidence | Broader regimes, fresh long-horizon tuning, particle convergence, more pairs and fixed-dataset replications |

Review: strongest alternative explanation is the single regime and limited particle control family; fresh regimes could reverse descriptive orderings. Independent Gaussian-sum and density-grid algebra plus exact short-T tests reduce shared-reference risk, but do not prove every long-T quadrature error bound. Local review with executable checks was used; no independent agent review.

Engineering, numerical validity and scientific interpretation remain separate: successful compilation is not accuracy; numerical reference agreement does not promote a particle method. No particle runs were repeated.

Correction GPU-owning wall charge 51.327502s; aggregate 6.795053/12 GPU hours; remaining 5.204947h. CPU-only checks/reporting add no GPU charge. All earlier pilots and the prior provisional hook reserve remain included.

Files: scores.csv contains every actual coordinate/reference/error; values.csv contains every likelihood; reference_convergence.csv preserves all resolutions; paired_differences.csv and heuristics.csv preserve the conditional calculations; cells.json and summary.json provide structured results. Original source, tuning and input hashes remain linked in report-manifest.json.
