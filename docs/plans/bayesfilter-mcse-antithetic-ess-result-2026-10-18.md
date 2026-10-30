# Antithetic ESS diagnosis: terminal result

18 October 2026, China time. This bounded diagnostic answers whether the
unavailable autocorrelation MCSE observed in the stationary stress study is a
BayesFilter translation defect or behavior inherited from the installed TFP
positive-pair estimator. It does not change tuning admission, posterior
criteria, the running SIR cohort, or any numerical default.

## Evidence and result

The actual comparison target was TensorFlow Probability 0.25.0's
`effective_sample_size` with `filter_beyond_positive_pairs=True` and
`cross_chain_dims=1`. We reconstructed 24 preserved cases: two replications,
three AR(1) correlations (`0`, `.99`, `-.99`), Gaussian and exponentiated
Gaussian margins, and 1,000 and 10,000 draws. Each case recorded the local
FFT implementation, installed TFP value, independent diagnostic equations,
positive-pair truncation, weighted denominator, raw variance estimate and
public availability. The first sixteen autocovariances were also checked by
direct dot products.

All source and input identities passed. The local implementation agrees with
the installed estimator to the inspected floating-point tolerance; the largest
relative errors were `4.849829993918111e-12` locally and
`3.367894855560023e-13` against TFP. Direct-lag checks passed. Three
strongly-antithetic Gaussian-coordinate cases have negative finite
denominators and negative ESS in all implementations. The public precision
route consequently reports an unavailable MCSE, rather than taking an
absolute value or silently switching to the separately identified monotone
ESS. This is the correct result for the named estimator's numerical domain.

The diagnosis therefore supports the following precise conclusion: the
claimed target, the BayesFilter computation and the installed TFP computation
are equal for the inspected finite program, while the estimator can be
unavailable for strongly antithetic paths. The diagnosis does not establish
that the estimator has calibrated finite-sample coverage, nor that the
monotone alternative is a valid replacement.

## Validation

Four regression tests preserve the negative-denominator examples, official
TFP parity, sign behavior and the public unavailable-MCSE contract. The
combined focused suite passed 35 tests, including the convergence and tuning
documentation contracts. The repaired official chapter was built in a fresh
directory with `latexmk`; the PDF has nine pages, and the rendered pages were
visually inspected. The only document repair was TeX escaping for a percent
literal and a repository path.

The CPU reference environment deliberately hid GPUs and recorded
`TF_FORCE_GPU_ALLOW_GROWTH=true`; NumPy was used only for the independent
diagnostic equations. No HMC sampling or tuning decision was made in this
phase.

## Decision and inference status

| Decision | Primary criterion | Veto/status | Next justified action |
| --- | --- | --- | --- |
| Close diagnosis | 24/24 cases, source hashes and direct-lag checks pass | No translation mismatch | Preserve cases and move to a separately planned estimator study |
| Keep current runtime semantics | Public unavailable status is honest for nonpositive variance | No implementation repair justified | Do not alter HMC or tuning behavior |
| Keep replacement/default open | Stress study found coverage failures and unavailable cases | No estimator passes a general calibration claim | Compare dependence-aware candidates on fresh controls |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No arithmetic or source-integrity veto; three negative-denominator cases remain explicitly unavailable |
| Statistically supported ranking | None |
| Descriptive evidence | ESS values, denominators and parity errors only |
| Default-readiness | Not established for any replacement |
| Next evidence needed | Exact finite-batch controls, fresh fixed/stopped calibration, skew-aware intervals and saved nonlinear-HMC quantities |

## Accounting and artifacts

The phase reserved 300 seconds from the reconciled SGQF balance. Measured
charges were 6.1850735219195485 seconds for the diagnosis,
10.005487486720085 seconds for focused tests, 2.4209604230709374 seconds for
the fresh LaTeX build and 2.283837 seconds for PDF rendering/inspection: total
`20.89535843171057` seconds. The unused `279.1046415682894` seconds are
released, leaving `134851.5684849512` seconds unallocated. SIR numerical caps
are unchanged. The exact commands and source hashes are preserved in
`antithetic-ess-20261018-01/{execution.json,checks-01.json,build-02.json,result.json}`;
the fresh PDF is under `documentation-build-02/diagnostics.pdf`.

## Remaining work

The next phase requires its own evidence contract and budget. It should compare
predeclared dependence-aware variance and batch policies against exact
covariance controls, then run fresh fixed-endpoint and stopped calibration on
independent seeds. It must include persistent, antithetic, tied-event and
skewed-square cases, preserve unavailable estimates, and never choose a policy
because it makes a failed HMC member pass. Only after that evidence should an
optional policy be evaluated on preserved full-model quantities and fresh
matched nonlinear/state-space models. No conclusion here rejects HMC or the
scientific direction; it rejects the unsupported claim that the current
precision procedures are generally calibrated.
