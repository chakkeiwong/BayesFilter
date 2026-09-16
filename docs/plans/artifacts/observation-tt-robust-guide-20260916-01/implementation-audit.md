# A10 mathematical claims and executable wiring

Executor review, not independent review. The optional extension preserves the
physical SV importance target. The eight manuscript propositions are analytical
statements under their stated assumptions, not empirical accuracy claims.

| Claim | Implementation | Focused executable evidence |
|---|---|---|
| Independent covariance scale detects node collapse | observation_robust_guide_tf.py:36,77 | test_independent_scale_guard_rejects_single_node_collapse |
| Signed cancellation and indefinite covariance differ | moment_kernel, checked_rule | test_signed_cancellation_and_indefiniteness_remain_separate |
| Convex SV mode has analytic gradient/Hessian | laplace_mode_kernel:134 | test_sv_mode_analytic_stationarity_and_positive_hessian, three observations |
| Positive quadrature uses the complete density ratio | repaired_update:173 | test_laplace_change_of_measure_recovers_predictive_gaussian |
| Chart changes preserve the physical guide joint | stable_charts:232; paired_gaussian; joint_sgqf_row_sampler | test_stable_chart_and_initializer_preserve_physical_joint |
| Mixture evaluates both densities at the actual draw | sample_physical_defense:325 | test_pair_density_evaluation_matches_sampler_and_mixture_both_branches |
| Normalization, exact importance identity, second-moment inequality | mixture_density_kernel:305 | test_mixture_normalization_importance_and_second_moment_by_quadrature |
| Actual filtering consumer uses the mixture | run_observation_aware_tt_complete.py:128 then update:93 | test_consumer_calls_physical_mixture_at_initial_and_later_steps |

The consumer forms previous log weight + log f + log g - log q, then uses
its existing normalized weights and declared ESS-triggered systematic
resampling. The mixture proof is not a claim of unbiased self-normalized means.
The initial step is an analytic SGQF Gaussian; subsequent TT fits have the
independent coordinate covariance, including the previous-state coordinate.

All 37 selected tests pass (tests-03.log), including unchanged neighboring
pair-block, SGQF-joint and warm-fit paths. New tests also check disjoint int32
seed blocks and calibration failure retention. GPU/XLA smoke-02 checks the
actual consumer in d1/d4, exposed failures and healthy fixtures.

No complete analytical parameter derivative, HMC endpoint, arbitrary-dimensional
quadrature, finite-particle ESS bound, or posterior accuracy certificate is
implemented or claimed here. The local mixture-score proposition is algebraic;
there is no claim that its formula differentiates adaptive selection, fitting,
resampling or the complete filtering program. Positive d<=4 tensor quadrature
is a guide diagnostic extension, not the historical all-axes retained-grid
Zhao-Cui production route.

The protected manuscript comparison has exactly one insertion (412 lines)
and no removed or replaced baseline lines before numerical closeout. The
MathDevMCP coverage and unresolved tooling limitations are in math-review.md.

The amplitude-sign addendum extends this audit: the actual fit_step endpoint
calls scale_initial_amplitude; tests-05 has 38 passes, including unchanged
positive-scale arithmetic, sign equivalence and zero-scale/norm rejection.
Smoke-03 replays the exact exposed data and fitting seed and accepts the
original -0.3106276392630666 scale at d4-s05 t19. Calibration-02's complete
selection ledger equals calibration-01. These are localized implementation
checks; fresh filtering confirmation remains a separate evidence requirement.

## Final confirmation and manuscript closeout

Confirmation-02 completes every TT arm on all 24 fresh cases, with zero TT
numerical/CDF/log-evidence failures. One d4 reference fails its precision
budget and is excluded from admitted accuracy comparisons for every arm.
The terminal checks verify disjoint sequence seeds, archived source hashes,
unchanged calibration selection and the physical density bound at all 1,920
mixture steps. Three unresolved guide updates explicitly use the predictive
fallback. The numerical repair passes its bounded checks; downstream quality
does not support promotion. The complete interpretation is in result-review.md
and the A10 result note, rather than inferred from passing implementation tests.

The final protected-manuscript comparison has exactly one 610-line insertion
and no removed or replaced baseline lines. Final LaTeX build and rendered-page
inspection pass. MathDevMCP coverage remains partial as documented in
math-review.md; no formal or total-gradient certificate is claimed.
