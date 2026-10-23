# Guided three-proposal mixture: engineering result, 16 October 2026

## Decision and scope

The bounded integration checks pass. Frozen quadratic and Gaussian-mixture
messages now enter the existing global and local maps in
`bayesfilter/highdim/covariance_proposal_tf.py`. The physical identity branch,
full mixture denominator, importance numerator and existing reset remain in
the same implementation. This is optional candidate functionality, not
canonical LEDH admission or evidence of improved model-score accuracy.

The earlier single-flow experiment used a different consumer. Its failed
candidates remain evidence about that consumer, not failure of this mixture.
Plan: `../../ledh-mixture-guide-integration-20261016.md`. Branch:
`ledh-refactor-with-policy-fix`; numerical HEAD:
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, plus recorded dirty sources.
All work used the LEDH execution checkout. Unrelated changes were preserved.
No commit, merge, push, environment mutation, training or research sweep.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain optional integration | Moment/density, zero-guide, derivative and wiring checks pass | Malformed precision rejects; tested accepted traces valid | Small synthetic fixtures | Plan fitted-guide comparison | Improved model-score accuracy |
| Preserve numerical policies | New checks explicitly use epsilon1, Sinkhorn5 | No silent numerical repair | Untuned baseline | Scope-specific tuning before claims | Universal optimality of (1,5) |
| Keep reset distinction | Existing mixture reset identified in code and manuscript | No canonical admission asserted | Canonical lifecycle equivalence not checked | Separate admission evidence if sought | Contract-E/UKF/TF32/HMC readiness |

## Implementation and mathematical audit

`make_filter` calls the shared `build_maps`. The guide changes the global
Gaussian-product moments and local information coefficients. Each particle
selects one map; positions are not averaged. The weight numerator remains
current likelihood times q0, and the denominator is the full density mixture.
Guide coefficients and beta are frozen; moving-anchor responsibilities, map
coefficients, densities, normalized weights and reset moments retain their
analytical tangents. The call-chain test executes the filter factory.
Zero-guide and one-component representation parity are checked.

The manuscript subsection `sec:bg-three-map-mixture` gives propositions and
proofs for guided maps, corrected weights/conditional variance, and total
differentiation, followed by pseudocode including OT and the separate reset.
Guided beta calibration must evaluate the guided maps; old unguided calibration
cannot silently transfer. A mixture guide supplies a local quadratic, not an
exact Gaussian-mixture product. The variance identity conditions on fixed
cloud/maps and independent proposal draws; it does not certify model-score
error or directly cover dependent SQMC draws.

The computed derivative is the total derivative of the returned finite
log-normalizer with fixed random inputs and coefficients. Central differences
check that same scalar and its particles, not the exact model likelihood.
MathDev verifies two scalar identities (global mean tangent and product
variance). Its automatic weight-proposition audit is **inconclusive** because
it extracted no obligation. This is neither a full proof verification nor a
failed proof. Complete responses are in `mathdev-audit.json`. The remaining
audit consists of the explicit derivations and bounded executable evidence.

## Run record

| Attempt | Scope | Result | Process wall seconds |
|---|---|---|---:|
| CPU01 | New guide tests and five existing checks | 12 passed | 25.52 |
| CPU02 | Bounded existing mixture suite | 12 passed, 8 deselected | 9.74 |
| GPU01 | Four-coordinate shared-consumer XLA fixture | All checks pass | 18.30 |

Five CPU tests overlap: **19 unique tests**, not24. Eight pre-existing
complete-recursive-score cases were not rerun. The new fixture uses two state
dimensions, three observations,24 particles and four parameter coordinates.
It checks quadratic/two-component guides and correction-off/on cases. CPU
commands hide GPUs before importing TensorFlow and are FP64 reference checks.

GPU01 maximum absolute score finite-difference error:7.258997847e-10; maximum
particle derivative error:1.807261445e-9. Maximum CPU-graph/GPU-XLA parity error
across recorded values/tangents:2.997602166e-15. All four checks have complete
valid three-step traces. The callable traced once; its log records XLA
compilation. These are fixture diagnostics, not statistical comparisons.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, TensorFlow
2.20.0-dev0+selfbuilt. GPU01 uses escalated access, FP64, XLA and TF32 off.
Actual device: **RTX5080**, recorded by TensorFlow. CUDA_VISIBLE_DEVICES=1 did
not select the less-busy card inferred from nvidia-smi: the enumeration orders
differed. Actual recorded hardware takes precedence over that initial intent.
Memory growth was verified before initialization; allocator peak333568 bytes.
This is not a performance comparison or production TF32 evidence.

`gpu01/manifest.json` records command, commit, source hashes, dirty diff,
guide/data, seeds, controls, device, memory policy, timing and plan/result links.
`gpu01/result.json` retains every check and trace. Seeds: [861,1], [861,2],
[861,3]; synthetic observations and guide are specified in runner/manifest.
CPU logs and process timings are `cpu01.log`, `cpu02.log`, `*.seconds`.

Exact numerical commands, from the LEDH execution checkout:

```bash
/usr/bin/time -f '%e' -o docs/plans/artifacts/ledh-mixture-guide-20261016/cpu01.seconds timeout 65s env CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_covariance_proposal_guide.py tests/highdim/test_covariance_proposal.py::test_beta_boundary_optimum_gap_and_simplex tests/highdim/test_covariance_proposal.py::test_beta_derivatives_convexity tests/highdim/test_covariance_proposal.py::test_maps_use_conditional_covariance_and_density_matches_sampler tests/highdim/test_covariance_proposal.py::test_master_calls_single_runtime_authority tests/highdim/test_covariance_proposal.py::test_runtime_call_chain_and_fail_closed > docs/plans/artifacts/ledh-mixture-guide-20261016/cpu01.log 2>&1
/usr/bin/time -f '%e' -o docs/plans/artifacts/ledh-mixture-guide-20261016/cpu02.seconds timeout 45s env CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_covariance_proposal.py -k 'not complete_recursive_score' > docs/plans/artifacts/ledh-mixture-guide-20261016/cpu02.log 2>&1
/usr/bin/time -f '%e' -o docs/plans/artifacts/ledh-mixture-guide-20261016/gpu01.seconds timeout 385s env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_mixture_guide_check.py --output docs/plans/artifacts/ledh-mixture-guide-20261016/gpu01 --budget-seconds 370 > docs/plans/artifacts/ledh-mixture-guide-20261016/gpu01.log 2>&1
```

Budget used:35.26 CPU/18.30 GPU process seconds plus one GPU second charged
for preflight. Remaining:44.74 CPU/380.70 GPU seconds of this repair allowance;
CPU attempts2/2, GPU1/2. Document builds are outside that numerical allowance.
`latexmk` was unavailable; pdflatex/bibtex built the manuscript instead. Logs
and rendered pages are under `latex01/`. Reader acceptance remains pending.
The final build (`pdflatex09.log`) has71 pages and no LaTeX warnings or
overfull/underfull boxes. Changed pages62–67 were visually inspected. The
review caught a malformed Section14.8 reference; it was repaired and the
rebuilt page66 was inspected again. Both source files pass the control-
character check. All runtime hashes still match the tested GPU manifest,
and `git diff --check` passes. Final PDF SHA256:
`d7d896e6bd832176d776e19b7b38386d71ce064d74a8023de9700c26301664e9`.

## Inference status and next action

| Evidence class | Status |
|---|---|
| Hard veto screen | No invalid accepted fixture output; malformed precision rejected |
| Statistically supported ranking | None; no accuracy campaign performed |
| Descriptive-only differences | Derivative discrepancies and fixture runtime only |
| Default-readiness | Not established; optional path and untuned development controls |
| Next evidence | Fitted-guide same-consumer comparisons with paired uncertainty |

Terminal red-team finding: a correct derivative can differentiate an inaccurate
approximation to the model likelihood. Synthetic guides do not establish
fitted-message quality, trained KDM integration or calibrated beta. Any later
apparent gain could come from changed geometry/calibration rather than useful
future information. Generalization beyond these Gaussian fixtures is the
weakest evidence. A density, moving-anchor derivative or reset counterexample
would overturn the corresponding implementation conclusion. Inferior model
scores would reject that candidate, not automatically this differentiation.

Next, plan fitted future messages and beta calibration through the SAME guided
consumer. Messages may condition on each evaluated sequence's future data,
using independent pilot randomness; coefficients stay frozen during parameter
evaluation. Settings selection and final comparisons use separate sequences.
Construct identity/bootstrap, defensive-global, defensive-local and equal-
mixture adversaries. Compare weak/persistent dynamics and mild/surprising
observations separately against exact Kalman model scores; include fitting
cost and paired uncertainty. This requires its own evidence contract and
budget, not repurposing the remaining mechanics allowance as promotion.
