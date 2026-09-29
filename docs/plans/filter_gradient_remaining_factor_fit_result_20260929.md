# Remaining factor-fit diagnostic result

The three saved unselected factor fits have accurate local objective gradients
at every tested initial and terminal state. Both CPU and GPU standalone and
instrumented owners reproduce the saved fits at the original complete-record
bounds. This unit finds no gradient defect at those states. It does not establish
convergence or explain the first divergent L-BFGS decision; the complete-record
differences remain preserved. No runtime equation, threshold, iteration cap,
selection rule, or source-policy allowance changed.

Baseline bebb2591a. Runs04923--04930 used92.795718 CPU and96.841738 GPU
process-seconds. GPU2 used UUID GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba
with verified memory growth; CPU was explicitly hidden-device reference work.
There were eight serialized workers, with one failed reader assertion04923:
the comparison operands were reversed relative to the saved authority.04924
corrected that reader orientation and reproduced all4534 existing differences.
No numerical kernel ran in the failed reader. Final04930 passes162 evidence and
policy checks; the guard remains315 sources/1457 existing exact allowances.

The source and executable saved-data audit binds4477 leaves to fits3/4/5,
and57 to selection/stability reporting. Neither factor family supplies the
selected geometry: factor1 has no accepted replicate and factor2 has only one;
direct selection requires two accepted stable replicates. Both backends use
the same dense consensus, with no selected-output/status/audit discrepancy at
the existing bounds. Unselected fits remain exported diagnostics.

| Fit | CPU/GPU termination | CPU/GPU terminal gradient infinity norm | Interpretation |
|---|---|---|---|
|3, factor1 replicate1 |200/200 iterations, unconverged, holdout rejected |0.00243127 /0.02385023 |Covariance condition about1.12e4; prediction-Jacobian condition about1e12 is a distinct identifiability diagnostic |
|4, factor2 replicate0 |7/7 iterations, optimizer failed, holdout rejected |30.35573 /30.35551 |Covariance condition about7444; failure is not evidence of severe covariance ill-conditioning |
|5, factor2 replicate1 |200/200 iterations, unconverged, fit gate accepted but unselected |0.05672699 /0.11019894 |Existing fit acceptance is not a convergence claim |

The geometry-fitting derivative is the existing TFP autodiff objective
gradient. It is not a filter analytical-score path.04925/04926 obtain the full
optimizer states from the ordinary factory and from the frozen factory with
two added outputs. All six instrumented/ordinary comparisons and all six
ordinary/saved-endpoint comparisons have zero differing fields at the original
bounds. Parameter-chart anchors agree across backends; initial raw parameter
differences are2.36e-12,8.69e-13 and6.10e-13 for fits3/4/5.

04927/04928 evaluate identical frozen CPU and GPU initial/terminal operands on
both backends:12 states per backend.04929 compares these with an independent
80-digit objective and manually derived covariance/chart pullback. For
L=sum(w_i ||P z_i-y_i||^2)/d, the reference uses
dL/dC=-P sym(dL/dP) P and includes the diagonal cancellation in
C=diag(s)[diag(1-row_norm(L)^2)+LL^T]diag(s). Both parameter charts pass
independent80-digit central differences at all coordinates on a nonuniform
weight control. Runtime binary64 constants are preserved in the reference.

Maximum gradient errors are7.82390e-12 CPU and1.89807e-12 GPU. Maximum ratios
to the original1e-10+1e-10*abs(reference) bound are0.07786 and0.01898.
Objective errors are at most1.78e-14. Same-backend replay gradients are exact
except one CPU discrepancy1.36e-16; the largest same-backend objective replay
error is7.11e-15. The terminal verifier also reconstructs the exact measured
source before/after a correction to reference constant rounding; the measured
TensorFlow objective callable was unchanged.

| Decision | Criterion/veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|
|Retain the existing objective |Independent local gradients and ordinary-output witnesses pass |The first optimizer trajectory divergence is not isolated |Reuse these states; if needed, trace the first divergent optimizer decision using output-preserving instrumentation |Unique solution or convergence |
|Preserve full-record discrepancy |4534 leaves still differ; selected geometry is unchanged |Exported unfinished fits have backend-sensitive trajectories |Keep separate numerical/status dispositions, without changing bounds |Whole-record equivalence or whole-program completion |
|Proceed to isotropic reporting |Rank-cut multiplicity makes the selected subspace non-unique |Current reports can expose arbitrary basis angles |Prepare an explicit refusal/reporting guard with well-separated-spectrum no-fire checks |Permission to choose another rank or regularize the matrix |

No performance ranking was attempted. This is deterministic diagnostic
evidence, not a posterior, HMC, canonical LEDH, or default-readiness result.
The strongest alternative explanation is a local numerical defect at an
untested intermediate line-search point; the checked initial/terminal
gradients cannot exclude that. A probe changing ordinary output cannot settle
that question. An ill-conditioned prediction Jacobian does not by itself
invalidate a well-conditioned covariance.

Artifacts are under docs/plans/artifacts/filter-gradient-repair-20260917 in the
main checkout. remaining-factor-04930-evidence.tar.gz contains34 reopened and
verified members,1805239 bytes, SHA256
17ba0b38756718ec36e4455a815bda80b932b38aedbfc15c9701c3359623c8b9.
The individual run manifests retain commands, environment, source hashes,
frozen input hashes and wall times. No independent agent review was used.
