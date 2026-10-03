# Public iAPF comparator: paper-conformance result

Completed 2026-09-20 under the [conformance plan](../../iapf-public-reference-paper-conformance-2026-09-19.md).

The pinned public R implementation passes the checked Gaussian twisted-filter
identities, but it is **not a full reference for the paper's stated outer
procedure and Section 5.1 fitting scheme**. Thirteen conformance checks pass;
four departures are recorded. All **49 focused regression tests pass**, including
tests that require rejection of the stronger reference label. Passing these
tests does not repair the public implementation.

## What was checked

The primary mathematical source is Guarniero, Johansen and Lee, *The Iterated
Auxiliary Particle Filter*: equations (5)--(6), Propositions 1--2, Algorithms
3--5, and Section 5.1 equations (15)--(16). The inspected local
[paper](../../../../.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf)
is hashed in the run manifest. The comparator is
[Sempreteamo/iAPF, iapf.R](../../../../.localresources/code/sempreteamo-iapf-a8811439/iapf.R),
commit `a88114395f6c11075fedc653db9480791b43391a`, SHA-256
`bf6f283b3eac0c3ef0a390af6be522af04026d47dcd6b1f52e606db6c5719851`.
Its original-author provenance remains **unverified**. The pinned file was not
modified.

The diagnostic executes parsed source function bodies. Independent scalar and
diagonal Gaussian calculations check initial and transition proposals, backward
normalizers, incremental weights, the telescoping path-density identity, and
recursive fitting targets. A nonideal three-time-step example exercises the
actual APF with and without resampling, including carried weights, ancestor
probabilities and accumulated likelihood. A separate ideal-twist example checks
Proposition 2. Controlled likelihood histories exercise the actual outer loop,
particle-count decisions and fresh final evaluation.

Base-R Gaussian density/sampling substitutes replace package dependencies; the
sampling basis is Cholesky rather than the original package's eigenbasis.
Prescribed innovations and ancestors isolate the weight arithmetic. At the
recursive-fit boundary, an optimizer substitute records actual target values
and supplies prescribed next-twist parameters. Only the outer-loop test replaces
APF/Psi with controlled responses. These substitutions are explicit in the
[R probe](../../../benchmarks/diagnose_iapf_public_reference.R); they establish
the checked calculations and control flow, not full dependency or RNG parity.
The source's top-level bootstrap/data-generation experiment was not executed.

## Departures from the paper

| Check | Executed source result | Paper target and implication |
|---|---|---|
| Fitting objective | Uses the inverse-scaled residual in `Psi`, lines 217--220 | Differs from equation (15), with parameter-dependent scaling; minimizers need not agree. |
| Positive floor | `psi_t`, lines 127--136, is a pure Gaussian | Equation (16) adds a positive constant, retaining a component of the original transition in the proposal. The source omits that protection. |
| Outer stopping | With `k=2` and constant likelihood history, stops at zero-based index 2 | Algorithm 4 permits stopping only for `l>k`, hence index 3. The source stops one likelihood estimate earlier in this fixture. |
| Single observation | Actual `APF` errors for `T=1` | The paper's filter has a well-defined one-observation case. R's `2:Time` loop is not empty when `Time=1`. This is an extension beyond the source's fixed `T=100` example, not a failure of that example. |

For the objective discrepancy, let `p` contain Gaussian densities at the fitting
particles and `y` the positive recursive targets. Write
\(a=p^Tp\), \(b=p^Ty\), and \(c=y^Ty\). Profiling the scale in equation (15)
gives

\[
\lambda=b/c,\qquad
L=\lVert p-\lambda y\rVert^2=a-b^2/c.
\]

The source computes this same scale but minimizes the other residual:

\[
F=\lVert y-p/\lambda\rVert^2
  =c^2a/b^2-c
  =\frac{c^2}{b^2}L.
\]

Because `b` depends on the Gaussian parameters, the multiplier is not constant
over the optimization. Equivalently, with \(R=L/a\),
\(F=cR/(1-R)\): this is a relative-shape objective. Executing the source objective
at three parameter points gives approximately `(0.0770023, 0, 0.0462168)`,
whereas the paper's profiled loss gives `(0.0665799, 0, 0.0336705)`. Agreement at
an exact fit cannot establish equality of the objectives away from that fit.

The objective and floor findings concern **fidelity to the paper's example
implementation**. Algorithm 3 permits other approximations, and a strictly
positive Gaussian can define a valid twisted filter. These findings therefore
do not prove that every likelihood estimate from this source is invalid, nor
that iAPF as a method fails. They do prohibit treating this source as an
unqualified implementation of the paper's full example procedure.

## Executable protection and validation

[The test suite](../../../../tests/highdim/test_iapf_public_reference_conformance.py)
checks known departures, missing/duplicate/drifted evidence, and the actual
comparison consumer. Three mutations of the executed R source respectively
double an initial weight factor, discard a carried weight, and delete the
terminal likelihood contribution. Mathematical checks detect each mutation,
independently of the source-hash mismatch.

The first test run exposed a fixture mistake: the ideal twist has unit terminal
weights, so deleting its zero log contribution cannot change the answer. The
terminal-contribution mutation now uses the nonideal retained-weight fixture.
The failed test log is preserved; this was a test-oracle repair, not a source
repair or a relaxed mathematical criterion.

The comparison's `run_iapf` now invokes the conformance check before its
TensorFlow comparisons. A deliberately damaged retained-weight result is
rejected at that consumer endpoint. Valid restricted comparisons carry
`reference_scope=matched_gaussian_operations_only` and
`paper_reference_eligible=false`, together with the full conformance report.
Self-stamping a passing eligibility flag cannot override failed requirements.

| Attempt | Result | Evidence |
|---|---|---|
| 01: standalone source audit | Complete; 13/17 checks pass; exit 2 means known nonconformance, not invalid execution | [Report](attempt01/report.json), [manifest](attempt01/manifest.json); 0.424 s |
| 02: new and directly affected CPU tests | 48 pass, one mutation-test oracle fails | [Log](tests-attempt02.log); pytest reports 33.78 s |
| 03: actual iAPF comparison consumer | Conformance report attached; 77 comparison checks pass and the known source fitting-accuracy check fails | [Results](comparison-attempt03/results.json), [log](comparison-attempt03.log); 8.402 s |
| 04: corrected focused CPU suite | **49 passed**, no skips | [Log](attempt04/pytest.log), [manifest](attempt04/manifest.json); 34.863 s subprocess wall time |

All commands used `/home/chakwong/anaconda3/envs/tftwogpu/bin/python` with
`CUDA_VISIBLE_DEVICES=-1`. Attempt 01 invoked
`docs/benchmarks/diagnose_iapf_paper_conformance.py --output docs/plans/artifacts/iapf-paper-conformance-20260919-01/attempt01`.
Attempt 03 invoked
`docs/benchmarks/diagnose_younis_iapf_kdm_reference_comparison.py --methods iapf --output docs/plans/artifacts/iapf-paper-conformance-20260919-01/comparison-attempt03`.
Attempts 02 and 04 invoked `-m pytest -q` with
`tests/highdim/test_iapf_public_reference_conformance.py`,
`tests/highdim/test_younis_iapf_relative_shape_tf.py`, and
`tests/highdim/test_younis_score_master_iapf_tf.py`.
Manifests preserve commit, commands, environment, fixtures, source hashes and
wall times; attempt 02's exact command and test time are preserved here and in
its log. Four of four attempts were used; conservatively charge 100/600 CPU wall
seconds including startup and overhead. No scientific particle campaign was
reopened and no GPU run occurred.

The consumer's remaining failure is `iapf/fit0/variance_source`: the public
optimizer returns approximately `0.9959796` for a variance known to be `1`,
outside the existing `0.002` tolerance. This is the previously recorded optimizer
quality finding, separate from paper conformance. The tolerance and public
source remain unchanged.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the public comparator only for the checked Gaussian operations | Pass for that scope | No mathematical failures in the specified core fixtures | General dimensions, horizons and dependency distributions are not exhaustively checked | Retain conformance checks at the consumer | Universal filter correctness |
| Reject full-paper reference status | Four deviations | Full-paper eligibility blocked | Original-author provenance remains unverified | Use paper-derived expectations; a stronger comparator must pass the missing procedure/example requirements | Failure of the published iAPF method |
| Keep the score-study interpretation qualified | No new score-performance evidence | Earlier promotion limits unchanged | Cause and true risk ordering for nonlinear case 1900 remain unresolved | Inspect the preserved first backward fit's coverage and geometry before a new scientific run | That these source departures caused the 1900 result |

| Inference category | Status |
|---|---|
| Hard veto screen | Full-paper comparator label rejected; known optimizer-accuracy failure preserved. |
| Statistically supported ranking | None requested or established by these deterministic checks. |
| Descriptive-only differences | Previous stochastic score comparisons retain their existing qualifications. |
| Default-readiness | No default, LEDH, HMC or runtime admission follows. |
| Next evidence needed | Paper-conforming procedure/fitting evidence for any stronger reference claim; separate target-specific diagnosis for case 1900. |

Post-run review: the strongest alternative interpretation is that this public
implementation intentionally uses a different Algorithm-3 approximation. That
is compatible with the restricted verdict, but cannot make its objective equal
to equation (15). The weakest coverage is the replaced numerical dependencies
and unexecuted top-level experiment; no claim is made about those paths. The
mutation-test repair also shows why exact cancellation cases must be paired with
nonideal examples. KDM conformance was not re-audited in this iAPF-specific task.
