# SIR moment-collapse repair: terminal result

20 October 2026; branch ledh-refactor-with-policy-fix; base commit
4d6915cc66e701c4c7b657e094306d298fdd3e76 plus preserved working changes.
Plan: ../../ledh-moment-collapse-repair-20261020.md.

## Finding and implemented repair

The failed comparison has three distinct layers. Severe importance-weight
concentration occurs before OT. Adding the resulting extremely small offsets
to a large state mean loses all physical spread in FP64. Finally, the moment
training endpoint collected, but did not enforce, the explicitly unavailable
materialized-shape flag. The preceding offset-preserving restoration repair
fixes its Cholesky failure; it does not recover representable particle spread.

This stage fixes the endpoint admission bug and propagates a covariance
residual relative to the actual target scale. The failed time4 cloud has
relative covariance loss1, not a harmless roundoff discrepancy. Baseline
particle states on both saved fixtures remain exactly unchanged. No ridge,
UKF lifecycle, OT count, training objective or scientific default was changed.

The existing moment_safety=True candidate was evaluated, not silently adopted.
It still loses all materialized spread on the failed fixture. That rejects
this protection as a repair for this failure; it is not a global rejection of
the safeguard or of simulated-moment learning.

## Upstream mechanism and scope

Call chain: Campaign.evaluate -> make_moment_endpoint ->
build_backward_guided_ledh_kernel -> _value_and_analytical_score_impl -> shared
flow, density correction and reset. Flow coefficients use predictive UKF P;
the actual zero-guide pre-flow draw and transition density use conditional Q.
Both densities and the flow Jacobian are accounted for. This is a poorly
matched, correctly importance-corrected proposal, not an omitted Jacobian.

At failed time4, mean trace(P)=1038.574 versus trace(Q)=18. The flow moves the
conditional mean32.812 state units, versus0.613 for the conditional reference.
Mean KL(actual conditional proposal || exact conditional posterior)=529.979;
ESS=1. The winning prior/proposal logit gap is69.784 and the observation
contribution is-0.808. A Q-flow reference on exactly the same incoming cloud
has ESS29.0407; the exact conditional proposal has ESS29.0408. These are
descriptive, conditional results, not a full-filter comparison or permission
to remove the UKF covariance lifecycle.

Shared-flow replay across both five-step fixtures agrees with saved children
within1.14e-13, proposal log densities within6.04e-14, and covariance carry
within3.46e-16 relative error. These small discrepancies pass engineering
parity; they are not treated as research blockers.

For the actual affine flow F(z)=b+Bz, a desired Gaussian N(m,V) has exact
preimage mean B^{-1}(m-b) and covariance B^{-1}VB^{-T}. Applying those samples
through the shared flow recovers the target samples within4.55e-13 on the
failed sequence. Its conditional weight identity holds within1.08e-12.
This is a checked fixed-cloud repair, not an integrated runtime guide.

The required additional information B^T V^{-1} B-Q^{-1} has minimum eigenvalue
-0.477 at time4. This excludes the trained positive-information quadratic
adapter. A terminal call-chain audit corrected an overbroad draft inference:
the campaign kernel arm actually selects mixture_product_moments, which
retains between-component covariance and can expand covariance. The sign
test does NOT prove that arm cannot represent the desired moments. Its
capacity for this particular preimage is not checked. A consumer regression
constructs a symmetric two-kernel example with variance2.75 from prior
variance1 and verifies the actual guided draw branch.

## Verification and provenance

- tests-02.log:102 passed in93.63 seconds, two dependency deprecation warnings.
- tests-03-mixture-call-chain.log:10 passed in7.45 seconds, same warnings.
- replay-01: GPU/XLA FP64, TF32 off, verified memory growth, RTX4080SUPER.
  OT(1,5), terminal balance32, K=N64, SIR d18/T5. Wall147.927113177 seconds.
- Dataset/particle seeds:262023017/274000034 and262023001/274000002.
  On the healthy safety-on fixture,6/6 analytical versus central-difference
  checks pass; maximum normalized error9.43374e-6 versus threshold1e-3.
  Healthy covariance residuals are below8.46e-13 for both arms. The failed
  fixture is rejected before derivative admission; no FD pass is claimed.
- mechanism-03: deliberately CPU-only, non-XLA reference;10 fixed-cloud rows,
  wall1.548927075 seconds. No NumPy runtime computation. Both diagnostic
  manifests preserve interpreter, command, hashes, data provenance and paths.
- mechanism-01 failed because TensorFlow solve required equal batch ranks;
  the explicit broadcast repair changed no input or scientific criterion.
  mechanism-02 then passed in1.61023546 seconds. Prior attempts are preserved.
- Manuscript: Section15.10 / Proposition45. Final two pdflatex passes succeed,
 94 pages, no final compiler warnings. Pages86--88 visually inspected. A
  transient TeX escaping error is preserved in latex-pass1.log and repaired;
  final logs are latex-final-pass1.log and latex-final-pass2.log.

Actual runtime commands are in replay-01/manifest.json and
mechanism-03/manifest.json; the Python environment is
/home/chakwong/anaconda3/envs/tftwogpu/bin/python. Replay used
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true with trusted GPU access.
All CPU tests and references used CUDA_VISIBLE_DEVICES=-1. Test command:

    python -m pytest -q tests/highdim/test_affine_restoration_offsets.py tests/highdim/test_ledh_unified_correction.py tests/highdim/test_higher_moment_contract_e.py tests/highdim/test_moment_safety_tf.py tests/highdim/test_moment_training_tf.py tests/highdim/test_moment_training_analysis_diagnostic.py tests/highdim/test_supervised_guidance_tf.py tests/highdim/test_ledh_backward_guide_tf.py
    python -m pytest -q tests/highdim/test_gaussian_mixture_conditioning_tf.py

Total stage GPU charge, including preceding restoration work:399.603107107
seconds/5 launches. Remaining:6800.396892893 seconds/3 launches. Inherited
campaign remainder:26862.354012593 GPU-process seconds. CPU tests/reference
time is separate. No training or independent comparison was resumed.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept endpoint/observability fixes | Rejection and unchanged-state checks pass | No regression found | Coverage beyond tested scope | Preserve fixes | Moment fidelity restored |
| Reject safety-on as collapse repair | Failed shape/covariance criterion | Time4 shape unavailable, relative loss1 | Other scopes | Address proposal geometry | Safeguard universally useless |
| Accept fixed-cloud preimage identity | Sample/density/weight identities pass | No replay mismatch | Integrated map/density tangents and learning | Plan shared-consumer flow-aware proposal | Full-filter improvement |
| Keep guide efficacy unresolved | Original comparison incomplete | Baseline numerical validity failed | Fresh-data moment risk and score error | Fresh validation after integration | Learned guides ranked or disproved |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Failed fixture still cannot supply materialized shape; safely rejected |
| Statistically supported ranking | None; two diagnostic fixtures are not independent efficacy replication |
| Descriptive-only differences | Conditional KL/ESS and the safety-on/off diagnostics |
| Default-readiness | Not evaluated; numerical controls and defaults unchanged |
| Next evidence needed | Integrated analytical flow-aware proposal, consumer parity, scope-specific calibration, fresh heldout moment comparison and heuristic checks |

## Terminal review and next action

Engineering: admission/covariance reporting tests pass. Numerical: healthy
analytical checks and exact reference identities pass, but the failed full
endpoint remains inadmissible. Scientific: no guide superiority or score
improvement established. The population simulated-label moment-risk identity
has not been invalidated by this implementation/numerical failure.

Post-run red-team: the strongest alternative explanation for the earlier
learning result is restricted training/capacity, not a failure of moment risk
as an objective. Fixed-parent conditioning is not the full nonlinear filter;
better one-step ESS alone cannot overturn that distinction. The negative
information eigenvalue constrains the quadratic adapter only. A successful
fresh full-filter moment comparison could overturn the current lack of
efficacy evidence. Two saved fixtures are the weakest empirical evidence.

Plan self-review preceded each diagnostic. Terminal math/code self-review
found and corrected the mixture-consumer inference and added its regression.
No independent mathdevmcp review was performed in this stage; visual and
self-review do not replace human readability review. Next is an explicitly
reviewed integrated flow-aware guide with total analytical map and density
derivatives, retaining UKF/Contract E/OT(1,5), before fresh efficacy testing.
