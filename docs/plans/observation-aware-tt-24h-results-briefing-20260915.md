# Completed results and additional 24-hour authorization

2026-09-15. Owner record H11: “you have 24 more hours for the campaign. What
are the problems? show me the results”. This adds 86400 seconds to the retained
816.529454-second H6 balance. The prior completed program and ledger are
preserved under `artifacts/observation-tt-continuation-24h-20260915-01/` before
this refresh. The new `budget.json` in that directory is the only ledger for
future spending; the old balance cannot also be spent through the old ledger.
The total ceiling includes design, review, implementation, computation and
reporting, charging concurrent work once and excluding idle/crash intervals.

This is an explanation and reporting step using completed evidence. It launches
no new experiment and changes no numerical control or promotion criterion.
The next numerical protocol will be a reviewed master amendment under the
existing owner authority. No further budget confirmation is needed within H11.

## What the filtering experiment shows

SGQF's approximate joint provides a useful filtering proposal. The conditional
is evaluated analytically from that Gaussian joint, with exact importance
correction. It is not the exact conditional distribution of the nonlinear
state-space model. The earlier current-marginal SGQF proposal is a different,
weaker comparator in these observations; both are included in the experiment.

The primary metric is squared error of the particle-filter mean against an
independent reference, divided by stationary state variance and averaged with
equal sequence weights. Lower values indicate smaller observed error. There
are 12 independent sequences per dimension, 20 observations per sequence,
four particle repetitions and 512 particles per method. All 24 reference
resolution/precision screens pass. The four-dimensional table uses the same
11 successful sequences for every method; the guide failure on the twelfth
sequence remains a veto and is not removed from the experiment's verdict.

| Proposal | 1D, all 12 sequences | 4D, matched 11 successful sequences |
| --- | ---: | ---: |
| Transition | .0021600 | .0066377 |
| Stationary prior | .0051131 | .0212687 |
| SGQF current marginal | .0037369 | .0060415 |
| SGQF joint conditional | .0018482 | .0029703 |
| Predictive-chart TT | .0015124 | .0072739 |
| Observation-guided TT | .0016963 | .0066699 |
| Original pair TT | .0016075 | .0026151 |
| TT with SGQF initialization and selection | .0016702 | .0023614 |

These are descriptive comparisons. No overall statistical ranking is established.
The old scalar TT methods were rerun alongside the new methods here; these are
not comparisons of incompatible metrics from different historical campaigns.

The [exported figure](artifacts/observation-tt-continuation-24h-20260915-01/completed-filtering-results.pdf)
is generated directly from the frozen A06 report. PNG and SVG copies, the
reporting script and source/output checksums are in the same directory. Its
axis scales differ between dimensions and it displays descriptive means only.

## Problems separated by cause

1. **TT conversion and fitting can lose accuracy already present in SGQF.** In
   A04's first four-dimensional transition with a Gaussian incoming law, the
   empirical squared Hellinger discrepancy is .001866 for the analytical SGQF
   joint, .003206 after polynomial/TT conversion without defense, .005384 after
   the then-inherited 5% defensive mixture, and .007671 after SGQF-started fitting.
   Generic-start fitting gives .009101. These are medians over three diagnostic
   row designs. Of the initializer's .002887 signed-amplitude projection error,
   .002850 is polynomial truncation and .0000371 rank compression. This example
   points first to polynomial representation and fitting, rather than assuming
   that increasing rank solves the problem. A05 separately calibrated the
   defensive mixture to 1e-5; A06 uses that calibrated value. The earlier 5%
   issue was addressed, but conversion and fitting reliability remain open.

2. **Validation selection is not a guaranteed SGQF performance floor.** A06
   keeps both TT starts and the unchanged analytical SGQF joint available.
   Validation therefore cannot select an empirical loss to SGQF on its own rows.
   On independent audit rows, however, the selection loses to SGQF in 9/228
   scalar and 1/209 four-dimensional transitions. In the successful 4D paths,
   SGQF-started TT is chosen 191 times, generic TT 10 times and exact SGQF eight
   times. Initialization is being used; selection generalization is unresolved.
   The audited density target contains the previous selected approximation,
   so better fitting of that target does not by itself establish a better true
   filtering distribution.

3. **The present signed-quadrature SGQF update can produce invalid covariance.**
   At zero-based sequence 8, time 18 in 4D, observation coordinates include
   -3.1916 and 2.7754 with observation scale .4. Every permitted quadrature
   level 2–5 has positive estimated mass but negative minimum covariance
   eigenvalue: approximately -6.68e-12, -.001137, -1.08550 and -.080229.
   A signed weighted sum of centered outer products does not guarantee a
   positive-semidefinite covariance, unlike a sum with nonnegative weights.
   This explains the structural vulnerability; the exact failure is reproduced
   by the unchanged CPU diagnostic. Higher levels did not monotonically repair
   this case. It invalidates six guide-dependent methods on that sequence.
   It does not establish that every SGQF implementation or Gaussian joint is poor.

4. **Observed filtering gains are neither universal nor yet reliably ranked.**
   In 1D ordinary observations, transition MSE is .0011964 versus .0014199 for
   the new TT. In 1D large observations, joint SGQF is .0011200 versus .0013987.
   The respective candidate-minus-comparator simultaneous 95% intervals are
   [-.00018662, .00063349] and [-.00050373, .00106115]. Both include zero:
   inferiority is not statistically established, but the predeclared positive
   observed-loss veto fires. No SGQF-joint contrast establishes an improvement.
   One stationary/large interval also misses the required precision. The
   complete 4D comparison cannot be ranked because of the missing guide sequence.

5. **Computational benefit and scalability remain unproved.** In this ordered,
   compiled-kernel-reusing campaign, median fit/total seconds per 4D sequence
   are .189/.665 for joint SGQF and 43.520/45.624 for the selected TT; shared
   guide construction is accounted separately. These descriptive times reveal
   substantial fitting work, not an equal-cold-start speed comparison. The
   full-coefficient initializer is restricted to d<=4 and is not a scalable
   high-dimensional implementation.

## Judgment and next amendment

The frozen A06 criterion requires complete valid comparisons, no heuristic
losses, adequate precision, and paired uncertainty supporting the required
non-inferiority against all declared simple proposals and observation regimes.
Its conservative rule rejects the current candidate. That is not a statistical
proof of inferiority, and it does not reject the TT research direction.

The next amendment must turn the two causal questions into separate tests:
obtain valid, accurate guide moments on the difficult observations; then preserve
the useful SGQF density during TT representation, optimization and selection.
Healthy-case non-harm and exposed-case localization can diagnose repairs. Fresh
calibration and untouched confirmation sequences are required before a new
filtering claim. The complete SGQF joint remains a comparator and available
proposal. The degree/rank/sweep/row choices must be justified and tested rather
than inherited as proven defaults. Cost must be reported alongside accuracy.
No phase number, algorithm repair, threshold relaxation or full launch is
silently activated by this briefing. The amendment will update the master before
execution and receive the independent review already requested by the owner.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain A04–A06 as completed, non-promoted evidence; record H11 budget | A06 advancement fails | d1 conditional heuristic losses; one d4 guide failure | Sequence-level uncertainty, guide robustness, fitting and selection generalization | Write and review the next master amendment under the added 24 hours | Universal SGQF failure, TT-direction rejection, reliable improvement, high-dimensional scalability |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current candidate fails heuristic and guide-validity requirements. |
| Statistically supported ranking | Seven narrow A06 d1 contrasts against transition/stationary/marginal-SGQF; no supported overall or joint-SGQF ranking. |
| Descriptive-only differences | The full table, TT comparisons, successful-guide d4 differences, fitting/audit counts and timing. |
| Default readiness | Not established. |
| Next evidence needed | Reviewed repairs, fresh target-specific calibration and complete independent confirmation with adequate uncertainty. |

The principal alternative explanation for favorable 4D numbers is conditioning
on the guide succeeding. The principal fitting limitation is that empirical
discrepancy to a recursively approximated target can improve while true filter
accuracy does not. Neither issue can be repaired by choosing nicer summary
statistics or deleting the failed sequence.

Evidence: [A04 fitting](observation-aware-tt-sgqf-initialization-20260915-result.md),
[A05 calibration](observation-aware-tt-defense-consumer-20260915-result.md),
[A06 filtering and intervals](observation-aware-tt-independent-filtering-20260915-result.md),
[completed-program closeout](observation-aware-tt-phase10-closeout-20260915.md).
