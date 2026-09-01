# Generic Complete-DMIS and TT-Control-Variate Repair

Date: 2026-09-01  
Status: completed for the bounded candidate/diagnostic scope; see
`docs/plans/bayesfilter-generic-dmis-control-variate-repair-execution-result-20260901.md`

## 1. Research intent

### Question

Can the high-dimensional squared-TT filter retain a model-independent,
analytical-gradient likelihood route when the fitted TT normalizer is a poor
estimate of the exact carried-density normalizer?

### Mechanism under test

For each filtering step, construct one or more normalized proposal densities
from any available guide (retained TT, UKF moments, LEDH, Gaussian, or Student),
combine them into a complete deterministic mixture, and evaluate the exact
finite carried target at the sampled points.  Optionally use the nonnegative
fitted squared TT only as a control variate with a known Gram integral:

\[
 \widehat Z_{\rm cv}=Z_H+
 \sum_{i=1}^N b_i\frac{\gamma(X_i)-h(X_i)}{q(X_i)}.
\]

The implementation must accept arbitrary component log densities and must not
contain a C2/model-name branch.  C2 is a holdout fixture only.

### Target boundary

The target is the finite carried-density quantity
\(Z_t^{\rm fin}=\int\gamma_t^{\rm fin}(x)\,dx\), where transition and
observation factors are exact and the previous density is the normalized
carried approximation.  This is not silently promoted to the true model
likelihood when the carried approximation is not exact.

### Primary criterion

On an analytic finite fixture, the complete-DMIS estimate and the control-
variate estimate must agree with the independently computed finite target
integral within the predeclared numerical tolerance.  If an iid replication
is added, its Monte Carlo interval is reported separately.  In all cases the
explicit directional derivative must agree with a finite difference of the
*same frozen value program*.

For a stochastic proposal comparison, a candidate is viable only when it is
finite, has complete-mixture weights, and does not fail the target/measure
identity.  An ESS increase is a nomination signal, not a correctness proof.

### Vetoes and explanatory diagnostics

Hard vetoes are: nonpositive or nonnormalised mixture/base masses, incomplete
mixture denominator, an unsupported target point for plain DMIS (or an
unsupported nonzero residual for the control-variate correction), nonfinite
value, nonpositive control-variate normalizer, failed derivative parity, or a
model-specific runtime fork.  The directional theorem additionally requires
positive complete-mixture density at every frozen row; a zero-density row is
masked and reported as `tangent_support_valid=false` rather than allowed to
produce a NaN.  ESS, maximum normalised weight, residual second moment, shell
error, and conditioning are explanatory diagnostics unless a later plan
explicitly promotes one.

No result from this plan establishes universal finite variance, exact posterior
inference, HMC readiness, or a new default.  A defensive component supplies a
pointwise variance bound only under its stated integrability assumption.

## 2. Mathematical contract

Let \(q_j\) be normalized component densities and let
\(\alpha_j>0\), \(\sum_j\alpha_j=1\).  The denominator is always

\[
 q(x)=\sum_{j=1}^J\alpha_jq_j(x).
\]

For a deterministic component bank with \(N_j\) rows from component \(j\),
use \(b_i=\alpha_j/N_j\); for an iid mixture bank use \(b_i=1/N\).  The base
masses must sum to one.  If \(q>0\) on the target support,

\[
 \mathbb E_q[\gamma(X)/q(X)]=\int\gamma(x)\,dx,
 \qquad
 \operatorname{Var}_q(\gamma/q)=\int\gamma^2/q-(\int\gamma)^2.
\]

For a nonnegative control \(h\) with known
\(Z_H=\int h\), define the residual estimator

\[
 \widehat Z_{\rm cv}=Z_H+\sum_i b_i r_i/q(X_i),
 \qquad r_i=\gamma(X_i)-h(X_i).
\]

The same cancellation proves unbiasedness for iid mixture draws.  Its variance
is \(N^{-1}\{\int(\gamma-h)^2/q-(\int\gamma-Z_H)^2\}\) for iid draws.  The
control variate is not required to be close for correctness; closeness is only
what can reduce variance.

For a frozen bank, let dots denote total directional derivatives of the values
at the frozen points (including pathwise terms if a map is intentionally
parameter dependent).  With
\(\dot\ell_q=\sum_j\rho_j\dot\ell_j\) and
\(\rho_j=\alpha_jq_j/q\),

\[
 \dot{\widehat Z}_{\rm cv}=\dot Z_H+
 \sum_i\frac{b_i}{q_i}
 \left[(\dot\gamma_i-\dot h_i)-r_i\dot\ell_{q,i}\right],
 \qquad
 \dot{\log\widehat Z}_{\rm cv}=\dot{\widehat Z}_{\rm cv}/\widehat Z_{\rm cv}.
\]

The mixture weights $\alpha_j$ and row masses $b_i$ are fixed in this kernel;
a route that differentiates either must add those terms explicitly.

Freezing the rows, proposals, ancestry, and discrete branch decisions gives a
different finite program from differentiating an adaptive sampler.  The plan
requires the choice to be recorded rather than hidden.

## 3. Assumption/default audit

| Choice | Provenance and role | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- |
| Exact component log densities | mathematical DMIS identity | a copied or selected-component density changes the denominator | recomposition and label permutation | required invariant |
| Positive mixture weights | defensive-support requirement | zero-weight component can leave target support uncovered | configuration validation | reviewed default |
| Base masses | deterministic-mixture identity | treating every stratified row as \(1/N\) biases the integral | mass sum and analytic fixture | required invariant |
| Squared TT \(h\) | existing retained Gram path | poor tail fit gives a large residual; cancellation can produce a nonpositive estimate | residual second moment and positivity flag | optional candidate |
| Frozen proposals/rows | analytical-gradient contract | omits adaptation derivatives if the route is described as adaptive | same-scalar finite difference | explicit hypothesis |
| Product Student defensive arm | tensor-product tractability | wrong degrees of freedom or scale gives weak overlap or divergent TT moments | density normalisation and tail shell check | proposal-only hypothesis |
| TensorFlow float64 candidate kernel | repository numerical backend | unsupported XLA op or accidental NumPy path | import, graph, and parity smoke | implementation choice |

The plan does not promote a Student reference measure.  A Student proposal and
a Student TT basis have different contracts; the latter requires new mass
matrices and contractions and is outside this repair.

## 4. Implementation scope

### 4.1 New generic module

Add `bayesfilter/highdim/frozen_dmis_control_variate_tf.py` with:

1. strict TensorFlow float64 validation of component weights and base masses;
2. `complete_mixture_log_density` using all components;
3. a value kernel for plain DMIS and the optional control variate;
4. an explicit tangent kernel accepting target, control, and component-log
   tangents; and
5. fixed-shape `tf.function` factories with `jit_compile=True` by default.

The module accepts tensors only.  It has no model imports, no model-name
conditionals, and no NumPy numerical path.  A caller supplies exact target
values/log densities and arbitrary proposal component log densities.

Returned diagnostics include the estimate, positivity/finite flags, complete
mixture log density, target-weight ESS, maximum normalized target weight,
residual second moment, and tangent values.  ESS is labelled descriptive.

### 4.2 Tests

Add `tests/highdim/test_frozen_dmis_control_variate_tf.py` covering:

- normalized Gaussian/Student component recomposition;
- component-label permutation invariance;
- deterministic base-mass closure;
- exact finite-sum DMIS identity;
- control-variate identity and a hand-computable residual-moment reduction
  case;
- explicit tangent versus central finite difference for target, control, and
  proposal dependence;
- fail-closed directional support behavior when a row has zero mixture density;
- failure on invalid masses and nonpositive control estimates; and
- compiled-kernel parity with the eager diagnostic kernel.

The tests use a small analytic target and are CPU-only diagnostics.  They test
the deterministic weighted-bank identity directly; iid unbiasedness remains
a proposition/Lean algebra boundary rather than a noisy unit-test pass/fail
criterion.  They do not claim C2 likelihood accuracy.

### 4.3 C2 execution

Run one bounded C2 smoke only after the generic unit tests pass.  Feed the
existing retained proposal and one fixed Student defensive component through
the generic denominator.  Preserve the C2 output under a fresh versioned
artifact directory.  The run is a compatibility diagnostic, not a promotion
run, and must not alter the existing C2 production route.

## 5. Execution stages and budget

* Stage A, document and symbolic checks: two LaTeX passes, one nominal
  exhaustive MathDevMCP label pass (the CLI's `--max-labels 0` mode), bounded
  focus batches if the assembler or derivation router cannot cover all labels,
  and one Lean finite-sum proof build.  The CLI pass and focus union are
  bounded to this document and produce a coverage audit, not a proof
  certificate.
* Stage B, implementation smoke: at most three focused pytest retries for
  localized harness repairs; no package installation or network fetch.
* Stage C, bounded C2 compatibility: one CPU smoke with at most two retries,
  fresh output paths, and no claim-bearing default change.

Stop interpretation immediately for a target/measure mismatch, failed exact
identity, failed tangent parity, missing required artifact, or accidental
model-specific call chain.  A low ESS or a failed C2 candidate is a repair
trigger, not evidence against the generic identity.

Each serious artifact records commit, command, environment, dtype/device,
seeds, wall time, plan path, and output hashes.  CPU-only runs set
`CUDA_VISIBLE_DEVICES=-1` before TensorFlow import and say so in the artifact.

## 6. Pre-mortem

| Misleading outcome | Discriminating check |
| --- | --- |
| Both estimators agree because the independent reference shares a bug | use a closed-form finite discrete sum and a second direct calculation |
| ESS appears high while the signed control residual is unstable | report target ESS separately from residual second moment and positivity |
| tangent parity passes only because proposals are accidentally frozen | perturb component log densities and include their supplied tangents |
| compiled and eager paths differ | same-input bit/relative parity at the callable boundary |
| C2 result is presented as a universal conclusion | artifact and result note label it holdout compatibility only |

## 7. Decision table

| Decision | Primary criterion | Veto | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Generic kernel correctness | closed-form identity and tangent parity pass | any invalid mass, support, finite, or parity check | repair implementation and rerun Stage B | variance efficiency |
| Control-variate viability | finite positive estimate and lower residual second moment on heldout fixture | cancellation/nonpositive estimate | use plain DMIS or tune control offline | universal variance reduction |
| C2 compatibility | exact target convention and finite complete weights | target/measure mismatch or nonfinite branch | preserve as diagnostic and inspect proposal tails | exact C2 likelihood |

## 8. Review record

The skeptical audit was completed before Stage B.  It explicitly verifies that
the baseline is the current complete-mixture route, that ESS is not a
correctness criterion, that support and positivity stop conditions are present,
that the gradient is a total derivative of the stated frozen program, and that
the implementation has no C2-specific fork.  It also records the deterministic
bank distinction: a finite weighted bank is checked against a closed-form
quadrature target, while iid unbiasedness is tested separately.  The audit
result is recorded in
`docs/plans/bayesfilter-generic-dmis-control-variate-repair-plan-review-20260901.md`.

The execution also found and repaired the zero-density directional edge case
described above.  The repair was localized to the generic kernel, covered by a
new regression test, and rerun before the C2 smoke; it does not alter the
finite target or the promotion criteria.
