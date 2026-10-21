# D4R4b finite-filter reference source audit

This audit compares the immutable `source-163d24b34` cost source with the
source bytes archived by the four independent references listed in the
[preconfirmation plan](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md).
It concerns the finite-filter posterior in native parameter coordinates.
It does not validate a learned transport, admit a sampler, or establish the
exact nonlinear state-space likelihood.

## Dependency argument

All original source bytes match their recorded hashes: 133 PP-SGQF, 132
PP-UKF, 133 SIR and 131 structural files. The four target implementations,
their dataset generators and the parameter charts match byte for byte.
`modern_nonlinear.py::make_nonlinear_comparison_target` (line 56) selects the
actual factory through the unchanged `modern_hmc_matrix` registry.
`FullStatusComparisonAdapter` delegates value and score to the target's
`neutra_batch_log_prob_and_grad_status`. It neither trains nor evaluates a
learned map. `modern_prepared_nonlinear.py::prepared_binding` uses a fixed
`PrecomputedMassArtifact`; the numerical target remains the same source
posterior. The saved factor is a warm-start hypothesis only.

The differences below are common to all four archives. The diagnostic checker
pins both digests of each reviewed difference. A subsequent change to either
side is unreviewed and rejected, rather than covered by a filename exclusion.

| Changed file and inspected definition | Effect and compatibility argument |
| --- | --- |
| `inference/__init__.py`, `__all__` and `_DIRECT_EXPORTS` | Adds lazy MAMS proposer names only. The unchanged `__getattr__` imports an export on demand; importing the target does not invoke the new proposer. |
| `hmc_candidate_set_artifacts.py::_validate_result_payload` | Adds duration-preserving repair-lineage validation. Neither the independent integrand nor direct target evaluation calls this artifact reader. |
| `hmc_candidate_set_execution.py::_source_closure` (line 233) | Adds the duration compiler to the tuning source inventory. It does not change target evaluation. |
| `hmc_candidate_set_tuning.py::{HMCControllerConfig,HMCRepairAction,HMCTuningCandidateSetController}` | Adds versioned duration repair and family evidence. These changes affect tuner scheduling and serialization, not filter values, scores, charts or reference quantities. |
| `neutra_transport.py::{NeuTraTransportConfig,NeuTraTransportTrainer}` | Changes diagnostic NAF configuration and a training-helper decomposition. No learned transport or optimizer is constructed by these target factories or by the fixed affine mass preparation. |
| `neutra_transport_core.py::{hoffman_masks,sigmoid_mixture,sigmoid_inverse,AutoregressiveStage,affine_mixture}` | Changes learned-network masks, inverse stopping and diagnostic affine/NAF operations. These operations are not called by the finite-filter integrands or the fixed mass transform. The inspected diff adds no target-changing top-level call. |
| `neutra_weighted_training.py` | Module documentation differs; executable AST is unchanged. |
| `tuning_contract.py::HMC_TUNING_INTERFACE_CAPABILITIES` | Changes descriptive trajectory-policy strings; target and numerical defaults are unchanged. |
| `experimental_batched_svd_sigma_point_tf.py::tf_batched_svd_sigma_point_value_and_score` (line 3616) | The archived version forwards `jit_compile`; the current committed version omits forwarding. This wrapper is actually called by UKF targets and cannot be dismissed as unused. The unchanged custom-gradient caller at `batched_svd_sigma_point_tf.py:55` omits the argument. Both the wrapper and its callee `tf_batched_svd_sigma_point_value_and_score_with_rule` default to `True` in both copies. Thus this actual route still uses `True`; the target calculation is unchanged. This argument does not extend to callers explicitly requesting `False`. |

The independent reference modules and their drivers are absent from the
committed cost snapshot. They remain in the checked archives and are used
from those archives by the parity diagnostic. Their absence is not permission
to use a replacement reference implementation. NumPy/SciPy remain independent
diagnostic backends. Their outputs never nominate or repair a tuning candidate.
The archive source walker is conservative over package imports and function
bodies; the explanations above establish which changed operations actually
affect the selected integrand. The current closure, including new dependencies,
is recorded as well as the archived one.

The archived independent scalar integrands use NumPy/SciPy factorization;
they do not use BayesFilter's compiled Sylvester operator. The current native
parity process does use the current native operator. Its binary and C++ source
are recorded in the new report, alongside the TensorFlow/NumPy/SciPy versions.
The original native-parity binary was not fingerprinted by the old source
walker; we cannot assert old/current binary equality. Compatibility rests on
the unchanged independent integrand and the inspected finite-filter equations,
supported by current native value/score parity. It is not an old binary claim.

## Numerical and quantity checks

For each target, evaluate the four original parity points and the four starts
prepared with the first rung's seed (5400 plus 10000 times the original model
index). The original SIR driver did not save points in `parity.json`; reconstruct
them exactly from its archived source: zero, the native center, and center
plus/minus `(.01,-.01,.01)`. The center is half the saved standardized proposal
center. The other archives save all four points explicitly.

Reuse the original full-horizon parity thresholds: maximum absolute value
error `1e-8`; maximum score error `abs(FD-score)/(1+abs(score)) <=2e-5`
at both `1e-4` and `1e-5` finite-difference steps. Their provenance is the
archived PP driver at lines 99–106, SIR at 75–88 and structural at 83–105.
They are inherited diagnostic tolerances, not posterior precision or global
correctness guarantees. Native status must also be valid without floors.

Check physical parameters, standardized coordinate squares and sigmoid upper
tail quantities against the archived definitions. PP and structural use
probit charts. SIR's native coordinates have prior SD `.5`; physical values
are `(.1,18,10)*exp(theta)`, and source quantities use `z=theta/.5`.
The archived SIR integration adds the constant `3*log(.5)` for that change of
variables. The native-density parity compares in theta coordinates, so adding
that Jacobian to the parity value would be wrong. The new quantity check uses
a convenience FP64 scaled roundoff tolerance `1e-12`; this can detect a wrong
chart but does not calibrate posterior error.

Recompute the eight-scramble means, RMS within-scramble SDs, randomization
MCSE and pointwise half widths from the actual final two rungs. The multiplier
`2.3646242515927844` is `scipy.stats.t.ppf(.975,7)` from the original eight-row
design. The summary reaggregation tolerance `1e-12` accommodates FP64 summation
order and is separate from the inherited `.002` relative reference threshold.
Both the between-rung mean change divided by final SD and the final pointwise
half width divided by SD must remain at most `.002` for every declared
quantity. The result remains empirically refined, with no rigorous integration
error bound or simultaneous-coverage guarantee.

## Skeptical implementation audit

The wrong-baseline risks are a dirty checkout in place of the committed
target, or a rewritten reference in place of the archived integrand. Both are
prevented by explicit roots and exact source inventories. A finite-point match
cannot waive an unknown numerical dependency; the source comparison rejects
one before parity. A quantity or data mismatch vetoes that target's posterior
comparison. Missing or duplicated scramble rows, altered recorded inputs,
nonfinite parity, and reference-summary inconsistencies also fail closed.
These vetoes do not stop the separate cost-only MAMS ladder. The diagnosis is
bounded by the existing 3600-second local phase allocation, including previous
tests. No posterior ranking or default change is made by this audit.

Regression checks cover all four quantity/horizon contracts, source mutation,
missing archive bytes, duplicate scrambles, NaN parity, wrong target/data/chart,
missing or misordered rungs, malformed summary values, and bridge/current-source
identity mismatches. Terminal evidence must distinguish a compatible reference
from a successful sampler; no compatibility verdict is recorded before the
current numerical checks complete.
