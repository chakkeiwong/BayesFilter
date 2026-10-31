# KSC frozen future-guidance diagnostic

No fitted controlled-SMC or KDM result; no canonical LEDH admission, default readiness, HMC readiness, or statistically supported ranking. Scores differentiate the same finite program with guide coefficients frozen.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action |
| --- | --- | --- | --- | --- |
| engineering_screen_passed | references, likelihood wiring, validity and two-step derivatives pass | no continuation veto; per-cell heuristic promotion vetoes recorded | Eight clouds per constructed situation | specify target-specific fitting protocol before training |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | no continuation veto; per-cell heuristic promotion vetoes recorded |
| Statistically supported ranking | None; no confirmatory uncertainty analysis |
| Descriptive-only differences | Conditional coordinate MSE and likelihood error |
| Default readiness | Not evaluated; untuned diagnostic controls |
| Next evidence needed | Target-specific fitting and untouched downstream score validation |

Guided arms lose 34 of 120 comparisons against constructed heuristics.
Every loss vetoes promotion of that arm in that situation and coordinate.

| Situation | Arm | Score MSE: persistence chart | Score MSE: log beta |
| --- | --- | ---: | ---: |
| ordinary | unguided_identity | 0.00040675685 | 0.075035308 |
| ordinary | unguided_global | 0.00030153504 | 0.01988064 |
| ordinary | unguided_local | 0.00022212672 | 0.026443349 |
| ordinary | unguided_equal | 0.00024884031 | 0.022368003 |
| ordinary | gaussian_future_global | 0.00024991956 | 0.028498019 |
| ordinary | gaussian_future_equal | 0.00024962042 | 0.021799402 |
| ordinary | mixture_future_global | 0.00017594389 | 0.032814625 |
| ordinary | mixture_future_equal | 0.00017352698 | 0.030221683 |
| ordinary | gaussian_observation_approximation | 0.0006936856 | 0.084316465 |
| negative_future_tail | unguided_identity | 0.00087366187 | 0.10162547 |
| negative_future_tail | unguided_global | 0.00051144012 | 0.035194701 |
| negative_future_tail | unguided_local | 0.00057051169 | 0.028779508 |
| negative_future_tail | unguided_equal | 0.00049650846 | 0.049496738 |
| negative_future_tail | gaussian_future_global | 0.00027128345 | 0.0458123 |
| negative_future_tail | gaussian_future_equal | 0.0001240644 | 0.054496831 |
| negative_future_tail | mixture_future_global | 0.00031142515 | 0.047758175 |
| negative_future_tail | mixture_future_equal | 0.00012555148 | 0.052486584 |
| negative_future_tail | gaussian_observation_approximation | 0.01451935 | 0.58618408 |
| positive_future_tail | unguided_identity | 0.79580598 | 24.619718 |
| positive_future_tail | unguided_global | 0.026558363 | 0.5520897 |
| positive_future_tail | unguided_local | 0.046079508 | 1.5684865 |
| positive_future_tail | unguided_equal | 0.10278855 | 4.1384509 |
| positive_future_tail | gaussian_future_global | 0.009373455 | 0.38825599 |
| positive_future_tail | gaussian_future_equal | 0.0662549 | 1.1491887 |
| positive_future_tail | mixture_future_global | 0.025092383 | 0.66854426 |
| positive_future_tail | mixture_future_equal | 0.05712213 | 2.9654996 |
| positive_future_tail | gaussian_observation_approximation | 0.12983622 | 0.8132826 |

The Gaussian observation approximation is a cheap heuristic with a different
likelihood. The enumerated seven-mixture likelihood remains the reference.

The claimed derivative is the analytical total derivative of the frozen-guide
finite estimator. Agreement with finite differences does not establish small
error against the exact likelihood score.

Strongest alternative explanation: proposal geometry, reset approximation, or
finite-cloud score error can dominate the quality of the future message.
A fitted guide passing untouched conditional downstream comparisons would
overturn candidate rejection. Eight clouds are the weakest statistical evidence.

Measured wall time: 40.983 seconds; CPU references: 0.865 seconds.
Full traces, reference checks, finite differences and source provenance are in
`result.json`, `evaluations.jsonl`, and `manifest.json` beside this note.
