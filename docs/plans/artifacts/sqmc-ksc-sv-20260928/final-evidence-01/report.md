# KSC mixture comparison

Program: FP64 TensorFlow/GPU/XLA diagnostic variant, TF32 off, Contract E and dual-cap safeguards on. This differs from the production FP32/TF32 target. Particle cells use independently selected and validated controls for their exact route/horizon; `scopes.json` links every tuning artifact. Gaussian Kalman is a moment-matched approximation. The mixture grid is a converged independent reference, not a mathematically exact long-horizon oracle. These results do not establish native-SV accuracy, HMC readiness, default readiness or broad method superiority.

Each final scope uses 1,008 particles and eight independent dataset/design pairs at theta=(1.5,0), with both score coordinates. The table reports mean score-vector L2 error, its sample standard deviation, and the standard error of its mean. Reference error is zero by definition.

| T | Method | Pairs | Mean L2 error | SD(error) | SE(mean error) | Mean absolute log-likelihood error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | Mixture reference | 8 | 0 | 0 | 0 | 0 |
| 10 | Gaussian Kalman approximation | 8 | 0.382443 | 0.362072 | 0.128012 | 0.552204 |
| 10 | IID | 8 | 0.213303 | 0.124946 | 0.0441752 | 0.116469 |
| 10 | Inverse CDF | 8 | 0.16295 | 0.121418 | 0.0429278 | 0.0965423 |
| 10 | Permutation .98 | 8 | 0.163598 | 0.120938 | 0.0427581 | 0.0967806 |
| 10 | Permutation .97 | 8 | 0.165039 | 0.12192 | 0.0431053 | 0.0972867 |
| 20 | Mixture reference | 8 | 0 | 0 | 0 | 0 |
| 20 | Gaussian Kalman approximation | 8 | 0.410783 | 0.267763 | 0.0946686 | 1.11431 |
| 20 | IID | 8 | 0.354792 | 0.245164 | 0.0866785 | 0.727044 |
| 20 | Inverse CDF | 8 | 0.276938 | 0.174712 | 0.0617701 | 0.209167 |
| 20 | Permutation .98 | 8 | 0.277205 | 0.175699 | 0.0621191 | 0.209312 |
| 20 | Permutation .97 | 8 | 0.276445 | 0.178403 | 0.0630749 | 0.210075 |
| 50 | Mixture reference | 8 | 0 | 0 | 0 | 0 |
| 50 | Gaussian Kalman approximation | 8 | 0.479412 | 0.228561 | 0.0808086 | 2.80882 |
| 50 | IID | 8 | 0.462673 | 0.254476 | 0.0899707 | 0.552222 |
| 50 | Inverse CDF | 8 | 0.568745 | 0.354564 | 0.125357 | 0.599697 |
| 50 | Permutation .98 | 8 | 0.574202 | 0.354111 | 0.125197 | 0.599474 |
| 50 | Permutation .97 | 8 | 0.571715 | 0.356339 | 0.125985 | 0.600678 |
| 120 | Mixture reference | 8 | 0 | 0 | 0 | 0 |
| 120 | Gaussian Kalman approximation | 8 | 0.49483 | 0.468102 | 0.165499 | 9.41472 |
| 120 | IID | 8 | 1.54362 | 1.15022 | 0.406663 | 1.68852 |
| 120 | Inverse CDF | 8 | 1.60816 | 1.24391 | 0.439788 | 1.28587 |
| 120 | Permutation .98 | 8 | 1.58895 | 1.22902 | 0.434523 | 1.28644 |
| 120 | Permutation .97 | 8 | 1.60149 | 1.22619 | 0.433522 | 1.2903 |

`scores.csv` contains every actual score coordinate, reference score and absolute error. `values.csv` contains every actual log likelihood and error. `summary.csv` also reports mean signed coordinate errors, coordinate SEs and the norm of the mean error vector; that norm differs from the mean of error norms.

The eight pairs vary both data and random design. Their SEs therefore are not estimates of conditional Monte Carlo uncertainty for one fixed dataset. Paired t intervals in `paired_differences.csv` are exploratory, use shared pairs, and have no adjustment for the many comparisons. Small-sample shape assumptions and limited model coverage constrain their interpretation.

## Actual mean scores and log likelihoods

Means below average the same eight datasets. Individual reference scores vary by dataset; use scores.csv for exact pair comparisons.

| T | Method | Mean gamma_raw score | Mean log_beta score | Mean absolute gamma error | Mean absolute log_beta error | Mean log likelihood |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10 | Mixture reference | -0.0804172 | 0.140762 | 0 | 0 | -23.6943 |
| 10 | Gaussian Kalman approximation | 0.0122538 | -0.0212964 | 0.122231 | 0.321316 | -23.9665 |
| 10 | IID | -0.140536 | 0.227131 | 0.11099 | 0.149527 | -23.7679 |
| 10 | Inverse CDF | -0.10595 | 0.201285 | 0.0852208 | 0.112389 | -23.7383 |
| 10 | Permutation .98 | -0.105016 | 0.202253 | 0.0842955 | 0.113631 | -23.7377 |
| 10 | Permutation .97 | -0.106094 | 0.202747 | 0.085921 | 0.114296 | -23.738 |
| 20 | Mixture reference | 0.710345 | -0.390211 | 0 | 0 | -46.5766 |
| 20 | Gaussian Kalman approximation | 0.57906 | -0.421943 | 0.199441 | 0.292392 | -47.6515 |
| 20 | IID | 0.707321 | -0.508053 | 0.186016 | 0.264378 | -47.2932 |
| 20 | Inverse CDF | 0.683561 | -0.415364 | 0.173993 | 0.175019 | -46.7787 |
| 20 | Permutation .98 | 0.687117 | -0.414753 | 0.173671 | 0.174468 | -46.7787 |
| 20 | Permutation .97 | 0.683199 | -0.416358 | 0.171037 | 0.175314 | -46.7794 |
| 50 | Mixture reference | -0.270517 | 0.0412347 | 0 | 0 | -117.055 |
| 50 | Gaussian Kalman approximation | -0.470321 | -0.0755826 | 0.354989 | 0.224205 | -119.863 |
| 50 | IID | -0.400291 | 0.114716 | 0.404372 | 0.176181 | -117.538 |
| 50 | Inverse CDF | -0.29746 | 0.0760522 | 0.50765 | 0.193826 | -117.218 |
| 50 | Permutation .98 | -0.296841 | 0.0771104 | 0.513651 | 0.193042 | -117.219 |
| 50 | Permutation .97 | -0.302615 | 0.0762956 | 0.510393 | 0.193548 | -117.219 |
| 120 | Mixture reference | 0.147717 | -0.758255 | 0 | 0 | -281.081 |
| 120 | Gaussian Kalman approximation | -0.169062 | -0.800216 | 0.351984 | 0.236848 | -290.496 |
| 120 | IID | -0.872229 | -0.524893 | 1.46626 | 0.370021 | -282.77 |
| 120 | Inverse CDF | -0.0859242 | -0.705323 | 1.50502 | 0.433788 | -282.367 |
| 120 | Permutation .98 | -0.111411 | -0.706264 | 1.46956 | 0.435781 | -282.368 |
| 120 | Permutation .97 | -0.102756 | -0.708148 | 1.49331 | 0.434315 | -282.372 |

## Exploratory paired error differences

Negative values favor the left method on these pairs. Each interval is a paired 95% t interval for the mean L2-error difference, without multiplicity correction. Intervals excluding zero support only that limited comparison under the t-interval assumptions; they do not establish a general ranking.

| T | Left | Right | Pairs | Mean difference | 95% interval |
| --- | --- | --- | ---: | ---: | --- |
| 10 | IID | Inverse CDF | 8 | 0.0503537 | [0.00966954, 0.091038] |
| 10 | IID | Permutation .98 | 8 | 0.0497055 | [0.0107897, 0.0886212] |
| 10 | IID | Permutation .97 | 8 | 0.0482641 | [0.00848189, 0.0880462] |
| 10 | Inverse CDF | Permutation .98 | 8 | -0.000648291 | [-0.00322852, 0.00193194] |
| 10 | Inverse CDF | Permutation .97 | 8 | -0.0020897 | [-0.00554926, 0.00136987] |
| 10 | Permutation .98 | Permutation .97 | 8 | -0.00144141 | [-0.00355081, 0.000668002] |
| 20 | IID | Inverse CDF | 8 | 0.0778537 | [-0.0882195, 0.243927] |
| 20 | IID | Permutation .98 | 8 | 0.0775874 | [-0.0866324, 0.241807] |
| 20 | IID | Permutation .97 | 8 | 0.0783468 | [-0.0837683, 0.240462] |
| 20 | Inverse CDF | Permutation .98 | 8 | -0.000266305 | [-0.00542479, 0.00489218] |
| 20 | Inverse CDF | Permutation .97 | 8 | 0.000493069 | [-0.00784526, 0.00883139] |
| 20 | Permutation .98 | Permutation .97 | 8 | 0.000759374 | [-0.00860867, 0.0101274] |
| 50 | IID | Inverse CDF | 8 | -0.106072 | [-0.351703, 0.139559] |
| 50 | IID | Permutation .98 | 8 | -0.111529 | [-0.354698, 0.13164] |
| 50 | IID | Permutation .97 | 8 | -0.109042 | [-0.353431, 0.135348] |
| 50 | Inverse CDF | Permutation .98 | 8 | -0.00545666 | [-0.0138641, 0.00295078] |
| 50 | Inverse CDF | Permutation .97 | 8 | -0.00296971 | [-0.0107956, 0.00485615] |
| 50 | Permutation .98 | Permutation .97 | 8 | 0.00248694 | [-0.00330239, 0.00827628] |
| 120 | IID | Inverse CDF | 8 | -0.0645378 | [-1.26033, 1.13125] |
| 120 | IID | Permutation .98 | 8 | -0.0453276 | [-1.22317, 1.13251] |
| 120 | IID | Permutation .97 | 8 | -0.0578728 | [-1.23996, 1.12421] |
| 120 | Inverse CDF | Permutation .98 | 8 | 0.0192102 | [-0.00285535, 0.0412758] |
| 120 | Inverse CDF | Permutation .97 | 8 | 0.00666506 | [-0.0148215, 0.0281516] |
| 120 | Permutation .98 | Permutation .97 | 8 | -0.0125452 | [-0.0284253, 0.00333503] |

## Conditional heuristic checks

Every observed loss below is a promotion veto for that case; it does not discard a finite candidate or stop the remaining comparisons. The IID method is an additional comparator for each SQMC route.

| T | Method | Heuristic | Observed losses / comparisons |
| --- | --- | --- | ---: |
| 10 | IID | first_only | 0/8 |
| 10 | IID | gaussian_kalman | 2/8 |
| 10 | IID | zero_score | 0/8 |
| 10 | Inverse CDF | first_only | 0/8 |
| 10 | Inverse CDF | gaussian_kalman | 2/8 |
| 10 | Inverse CDF | iid | 1/8 |
| 10 | Inverse CDF | zero_score | 0/8 |
| 10 | Permutation .98 | first_only | 0/8 |
| 10 | Permutation .98 | gaussian_kalman | 2/8 |
| 10 | Permutation .98 | iid | 1/8 |
| 10 | Permutation .98 | zero_score | 0/8 |
| 10 | Permutation .97 | first_only | 0/8 |
| 10 | Permutation .97 | gaussian_kalman | 2/8 |
| 10 | Permutation .97 | iid | 1/8 |
| 10 | Permutation .97 | zero_score | 0/8 |
| 20 | IID | first_only | 1/8 |
| 20 | IID | gaussian_kalman | 2/8 |
| 20 | IID | zero_score | 0/8 |
| 20 | Inverse CDF | first_only | 1/8 |
| 20 | Inverse CDF | gaussian_kalman | 2/8 |
| 20 | Inverse CDF | iid | 2/8 |
| 20 | Inverse CDF | zero_score | 0/8 |
| 20 | Permutation .98 | first_only | 1/8 |
| 20 | Permutation .98 | gaussian_kalman | 1/8 |
| 20 | Permutation .98 | iid | 2/8 |
| 20 | Permutation .98 | zero_score | 0/8 |
| 20 | Permutation .97 | first_only | 1/8 |
| 20 | Permutation .97 | gaussian_kalman | 1/8 |
| 20 | Permutation .97 | iid | 2/8 |
| 20 | Permutation .97 | zero_score | 0/8 |
| 50 | IID | first_only | 1/8 |
| 50 | IID | gaussian_kalman | 4/8 |
| 50 | IID | zero_score | 0/8 |
| 50 | Inverse CDF | first_only | 0/8 |
| 50 | Inverse CDF | gaussian_kalman | 6/8 |
| 50 | Inverse CDF | iid | 6/8 |
| 50 | Inverse CDF | zero_score | 0/8 |
| 50 | Permutation .98 | first_only | 0/8 |
| 50 | Permutation .98 | gaussian_kalman | 6/8 |
| 50 | Permutation .98 | iid | 6/8 |
| 50 | Permutation .98 | zero_score | 0/8 |
| 50 | Permutation .97 | first_only | 0/8 |
| 50 | Permutation .97 | gaussian_kalman | 6/8 |
| 50 | Permutation .97 | iid | 6/8 |
| 50 | Permutation .97 | zero_score | 0/8 |
| 120 | IID | first_only | 2/8 |
| 120 | IID | gaussian_kalman | 6/8 |
| 120 | IID | zero_score | 1/8 |
| 120 | Inverse CDF | first_only | 1/8 |
| 120 | Inverse CDF | gaussian_kalman | 6/8 |
| 120 | Inverse CDF | iid | 4/8 |
| 120 | Inverse CDF | zero_score | 1/8 |
| 120 | Permutation .98 | first_only | 1/8 |
| 120 | Permutation .98 | gaussian_kalman | 6/8 |
| 120 | Permutation .98 | iid | 4/8 |
| 120 | Permutation .98 | zero_score | 1/8 |
| 120 | Permutation .97 | first_only | 1/8 |
| 120 | Permutation .97 | gaussian_kalman | 6/8 |
| 120 | Permutation .97 | iid | 4/8 |
| 120 | Permutation .97 | zero_score | 1/8 |

## Terminal review

Completed scopes: 16/16. Valid final cells: 128/128; expected final cells: 128. Failed attempts and candidate rejections are preserved separately in terminal-review.json. Source snapshots, shared observations/reference scores, seed separation, actual-coordinate errors and quadrature convergence were checked during assembly.

Largest observed quadrature discrepancies: value 5.11591e-13, score coordinate 4.60743e-14. Refinement agreement is not a rigorous bound on quadrature error.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain all four for research | Owner decision; report actual errors | Apply per-case heuristic and validity vetoes | Single regime, eight pairs, limited controls | Broader regimes and replication before ranking | No default promotion |
| Accept completed comparisons as bounded evidence | Shared target and converged references checked | Invalid cells and reference failures remain explicit | Finite quadrature and particle/flow bias | Inspect paired intervals and sensitivity records | No native-SV equivalence |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | 0 scope failures; 0 nonzero attempts; 113 conditional heuristic losses |
| Statistically supported ranking | Only the predeclared exploratory pairwise intervals are available; no overall ranking or familywise claim |
| Descriptive-only differences | Observed mean errors, likelihood errors, sensitivity changes and runtimes |
| Default readiness | Not established |
| Next evidence | Independent replication, broader parameter regimes, N/resolution convergence and scope-specific safeguard calibration |

Strongest alternative explanation: a favorable result can arise from the single persistent regime, limited per-scope control search, or a few dataset/design draws. A reversal under broader regimes or fresh pairs would overturn a general ranking. The mixture approximates native SV, so matching this reference does not establish exact native-SV scores. Sensitivity diagnostics record changes without selecting settings by final accuracy.

## Compute accounting

Total conservative charge: 6.780795 of 12 GPU-hours, including prior work and the 300-second provisional commit-hook reserve. Remaining: 5.219205 GPU-hours. Renewed elapsed deadline: 2026-09-30T16:15:40.010888+00:00. No package change, HMC, push, publication or scientific/default promotion was performed.
