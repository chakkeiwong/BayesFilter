# Fable Audit Handoff: Direct-Factor SR-UKF Plan

Date: 2026-08-15
Status: `AUDIT_REQUESTED_NO_IMPLEMENTATION_AUTHORITY`
Requestor: BayesFilter
Reviewer: Fable, using the local MathDevMCP CLI
Primary plan: `docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md`

## 1. Review objective

Audit the execution plan and its mathematical derivation before implementation.
The review must determine whether the proposed direct-factor SR-UKF is:

1. mathematically sound under the stated sigma-point and lower-factor
   conventions;
2. numerically stable under finite precision, QR sign choices, triangular
   solves, and sequential Cholesky downdates;
3. robust to rank loss, failed pivots, nonfinite values, parameter-dependent
   observation noise, nonlinear center residuals, static batch composition, and
   XLA execution; and
4. covered by adequate primitive unit tests and end-to-end integration tests.

This is a bounded, read-only audit. Fable must not edit BayesFilter or
MacroFinance files, change defaults, run NeuTra/HMC, install packages, fetch
network resources, or approve a production switch. MathDevMCP output is
diagnostic review evidence. It is not by itself a proof certificate, numerical
qualification, or scientific promotion decision.

## 2. Source boundary and provenance

Read exactly these paths first:

```text
/home/ubuntu/workspace/BayesFilter/docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md
/home/ubuntu/workspace/BayesFilter/bayesfilter/linear/qr_factor_tf.py
/home/ubuntu/workspace/BayesFilter/bayesfilter/linear/kalman_qr_derivatives_tf.py
/home/ubuntu/workspace/BayesFilter/bayesfilter/nonlinear/srukf_factor_tf.py
/home/ubuntu/workspace/BayesFilter/bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py
/home/ubuntu/workspace/BayesFilter/bayesfilter/nonlinear/srukf_route_guard.py
/home/ubuntu/workspace/BayesFilter/tests/test_srukf_factor_tf.py
/home/ubuntu/workspace/MacroFinance/docs/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex
/home/ubuntu/workspace/MacroFinance/docs/plans/two_currency_double_zlb_dz5_bayesfilter_direct_factor_srukf_handoff_2026_08_15.md
```

The active BayesFilter commit at handoff is
`3030d86df9cb00346df82c7c19f015c09c7c6e1f`. Record the commit and SHA-256 of
every reviewed source file. Existing unrelated dirty files are out of scope.

The source anchor for the monograph algorithm is:

```text
MacroFinance/docs/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex
label: alg:nf_sr_ukf
label: rem:srukf_ad_design
```

The plan intentionally replaces the monograph's deliberate covariance
refactorization deviation with a direct downdate route. That is a planned
algorithmic repair, not a claim that the current implementation already follows
the monograph.

## 3. Required review questions

Fable must answer each question with one of:

```text
SUPPORTED       the supplied derivation/plan is internally justified;
REVISE          a concrete gap or incorrect step must be repaired;
ABSTAIN         the selected MathDevMCP/tool path cannot decide the question;
COUNTEREXAMPLE  a bounded counterexample or failure mode was found.
```

### 3.1 Mathematical soundness

1. Does `S_aug = block_diag(S_x,S_q)` represent the intended independent joint
   state/innovation covariance, and are its derivatives correctly represented?
2. Given `points = mean + offsets @ S'`, are the predicted and innovation QR
   stacks oriented so that `S S' = sum w_i dx_i dx_i'` and
   `S_y S_y' = sum w_i dy_i dy_i' + R`?
3. Is the exact DZ5 rule correctly stated for `alpha=1,beta=2,kappa=0`,
   including `w0_m=0`, `w0_c=2`, and noncentral weights `1/(2d)`?
4. Is the center residual correctly included after nonlinear propagation?
5. Is process noise included exactly once, with no accidental second `Q` append?
6. Are the positive-diagonal QR derivative identities correct on a fixed sign
   branch, including the transpose from upper `R` to lower `S`?
7. Does the gain formula use the same `P_xy` and `S_y` as the likelihood, and
   are the triangular-solve orientations dimensionally correct?
8. Is the factor-native likelihood derivative equivalent to the covariance-form
   score, including the sign `d innovation = -d observation_mean`?
9. Does the sequential downdate produce exactly
   `S_f S_f' = S_pred S_pred' - K S_y S_y' K'`?
10. Are the scalar downdate derivative equations correct, particularly the use
    of pre-update `a,u` values and the quotient derivatives for `c,s`?
11. Are the assumptions for every identity explicit: SPD factors, positive QR
    pivots, fixed sign branch, finite derivatives, and positive downdate margin?
12. Does any stated parity claim accidentally imply general nonlinear equality
    between a lower-QR factor orientation and a symmetric principal-root
    orientation?

### 3.2 Numerical stability and robustness

1. What happens as a QR pivot approaches zero? Is failure detected before a
   sign branch or derivative division becomes invalid?
2. What happens as a downdate margin approaches zero? Is the route rejected
   without adding an undocumented nugget?
3. Can sequential downdates accumulate more rounding error than a block method?
   Should the plan add a conditioning diagnostic or a bounded re-orthogonalizing
   alternative for comparison, without using it as a hidden fallback?
4. Are factor diagonals, residual norms, solves, log determinants, score
   increments, and derivative tensors checked for finiteness at every step?
5. Is the positive-diagonal QR sign normalization deterministic across CPU/GPU
   and eager/graph/XLA modes?
6. Does the derivative route remain valid when (R(\theta)) changes while the
   observation residual stack also changes?
7. Are static shape requirements and batch-native execution strong enough to
   prevent row-dependent behavior caused by batch shape or composition?
8. Are all failure cases distinguishable in diagnostics: QR failure, downdate
   failure, nonfinite derivative, invalid observation factor, and batch contract
   violation?
9. Does the plan distinguish a mathematical SPD failure from an implementation
   roundoff failure, and preserve the original evidence for both?
10. Are GPU memory growth, dtype, TF32/XLA state, device identity, and execution
    mode recorded sufficiently for numerical reproducibility?

### 3.3 Testing completeness

1. Do primitive tests cover value reconstruction, derivative reconstruction,
   finite differences, shape errors, dtype errors, nonfinite inputs, and failed
   downdates?
2. Does the one-step suite include nonzero process and observation noise,
   parameter-dependent (R), exact DZ5 weights, and a nonzero center residual?
3. Does the multi-step suite verify that factors, not covariances, are the
   carried state authority?
4. Is there an independent comparator that is not the implementation under
   test for score and covariance identities?
5. Does the suite test duplicate, mixed, permuted, and production-sized batches?
6. Does it test both scalar/batch-one and batched execution against the same
   physical row?
7. Are CPU eager, CPU graph, CPU XLA, trusted-GPU eager, graph, and XLA modes
   either qualified or explicitly classified unsupported with artifacts?
8. Does the static route guard reject eigh, SVD, principal-root, covariance
   refactorization, and hidden fallback references?
9. Are finite-difference steps, tolerance selection, condition diagnostics, and
   repeatability recorded before rare-row qualification?
10. Are integration tests tied to artifact schemas and source hashes rather than
    only checking a finite scalar output?

## 4. MathDevMCP CLI protocol

Run from `/home/ubuntu/workspace/BayesFilter` with the local checkout. Use no
network or package mutation:

```bash
export MATHDEVMCP_ROOT=/home/ubuntu/workspace/MathDevMCP
export PYTHONPATH="$MATHDEVMCP_ROOT/src"
```

If the shell does not permit persistent variables, prefix each command with
`PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src`. All outputs must be copied
to a unique audit artifact directory such as
`docs/plans/artifacts/direct-factor-srukf-fable-audit-20260815/` without
overwriting an existing result.

### 4.1 Environment and source hashes

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli doctor
git rev-parse HEAD
sha256sum \
  docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md \
  bayesfilter/linear/qr_factor_tf.py \
  bayesfilter/linear/kalman_qr_derivatives_tf.py \
  bayesfilter/nonlinear/srukf_factor_tf.py \
  bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py \
  bayesfilter/nonlinear/srukf_route_guard.py \
  tests/test_srukf_factor_tf.py \
  /home/ubuntu/workspace/MacroFinance/docs/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex
```

`doctor` and hashes establish provenance only. They do not certify the plan.

### 4.2 Source-bound monograph lookup

Use the exact monograph root and file; do not search the whole filesystem:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli search-latex \
  "Square-Root UKF QR Cholesky downdate" \
  --root /home/ubuntu/workspace/MacroFinance/docs/latex-papers/CIP_monograph \
  --file chapters/ch17_nonlinear_filtering.tex --limit 20
```

If a local LaTeX index is useful, build it under `/tmp` or the unique artifact
root. The lookup is source navigation, not proof of the paper's correctness.

### 4.3 Assumption inventory

Run one question per target identity:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli assumptions-for \
  "S S^T = A A^T for S = transpose(R), A^T = Q R"

PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli assumptions-for \
  "L_new L_new^T = L L^T - v v^T under a sequential Cholesky downdate"

PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli assumptions-for \
  "d logdet(S S^T) = 2 trace(S^{-1} dS)"
```

Record which assumptions are mathematical requirements and which are only
implementation diagnostics. Do not pass prose in `--given` as a formal proof
assumption; use explicit assumptions only when a symbolic route supports them.

### 4.4 Bounded identity/refutation checks

These are diagnostic symbolic checks. They should be run separately so a failed
identity is attributable to one obligation:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli derive-or-refute \
  "transpose(R) R = A transpose(A)" \
  --given "A transpose = Q R" \
  --assumption "Q transpose Q = I"

PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli derive-or-refute \
  "d(S S^T) = dS S^T + S dS^T"

PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli derive-or-refute \
  "d(z^T z) = 2 z^T dz"
```

If the parser cannot represent matrix dimensions or noncommutative products,
record `ABSTAIN` and perform the same obligation through the human derivation
and numerical tests. Never convert a parser simplification into a matrix proof.

### 4.5 Math-to-code structure checks

Run the structural checks against one exact file at a time:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli audit-kalman-recursion \
  bayesfilter/linear/kalman_qr_tf.py \
  --required-operation qr \
  --required-operation triangular_solve

PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli audit-math-to-code \
  "S_f S_f^T = S_pred S_pred^T - K S_y S_y^T K^T" \
  bayesfilter/nonlinear/srukf_factor_tf.py

PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli code-implements-equation \
  "logdet(P_y) = 2 sum(log(diag(S_y)))" \
  bayesfilter/nonlinear/srukf_factor_tf.py
```

Before implementation, the last two checks are expected to report missing or
partial coverage because the current prototype is known to refactorize the
filtered covariance and omit explicit observation noise. That is useful
baseline evidence, not a failure of this plan. Re-run them after implementation
against the new factor backend.

### 4.6 Test generation request

Ask MathDevMCP for diagnostic test obligations, then compare them with the plan's
test matrix:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli generate-math-tests \
  "S_f S_f^T = S_pred S_pred^T - K S_y S_y^T K^T" \
  --assumptions '["S_pred is lower triangular with positive diagonal", "S_y is lower triangular with positive diagonal", "all downdate margins are strictly positive"]' \
  --kinds '["symbolic_identity", "finite_difference", "random_spd", "near_singular", "failure_case"]' \
  --expected-failure-mode "fail closed on nonpositive downdate margin"
```

The generated cases are nominations. Fable must inspect whether they include
batch permutations, parameter derivatives, center residuals, observation-noise
derivatives, and XLA/eager parity; add missing obligations to the audit report.

### 4.7 Review packet

Build one compact packet after collecting the above outputs:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src python -m mathdevmcp.cli prepare-review-packet \
  "Is the BayesFilter direct-factor SR-UKF execution plan mathematically sound, numerically stable, robust, and adequately tested?" \
  --source '{"plan":"docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md","monograph":"/home/ubuntu/workspace/MacroFinance/docs/latex-papers/CIP_monograph/chapters/ch17_nonlinear_filtering.tex","commit":"3030d86df9cb00346df82c7c19f015c09c7c6e1f"}' \
  --evidence '[]' \
  --packet-id bayesfilter-direct-factor-srukf-20260815 \
  --handoff
```

The packet must preserve evidence classes, assumptions, abstentions,
counterexamples, and nonclaims. It must not be used as an approval token.

## 5. Human audit checklist

Fable's final report must contain the following sections:

### A. Derivation verdict

For each of the eleven mathematical questions in Section 3.1, cite the exact
plan subsection and either confirm the identity, identify a missing assumption,
or provide a counterexample. Pay special attention to:

- row-offset versus column-offset sigma-point orientation;
- lower factor versus upper QR factor transposes;
- center residual inclusion with `w0_c=2`;
- process noise exactly once;
- parameter-dependent `S_r` and `dS_r`;
- gain solve orientation;
- score sign and factor log determinant;
- sequential downdate equivalence;
- derivative propagation through the downdate rotations.

### B. Stability verdict

List every division or square root and its required guard. State whether a
positive diagonal/pivot or downdate margin is sufficient, and identify any
additional conditioning diagnostic required. Distinguish a mathematically
indefinite covariance from finite-precision loss of a positive margin.

### C. Robustness verdict

Audit static shapes, batch invariance, duplicate rows, mixed rows, permutations,
parameter derivatives, lagged observations, XLA compilation, GPU/CPU parity,
and artifact provenance. Identify any hidden fallback or source path that could
reintroduce per-step eigh/SVD/refactorization.

### D. Test verdict

Map every plan acceptance criterion to at least one unit or integration test.
Mark missing independent oracles, missing failure cases, missing execution
modes, and tests that only verify finiteness without verifying the mathematical
identity.

### E. Required changes

For every `REVISE` or `COUNTEREXAMPLE`, give:

```text
issue_id
severity: blocking | material | advisory
exact source/plan path
failure mechanism or counterexample
required repair
test that closes the issue
```

### F. Final decision

End exactly with one of:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```

`AGREE` means only that the written plan is sufficiently specified for bounded
implementation to begin. It does not approve implementation completion,
MacroFinance integration, NeuTra, HMC, or production use. `REVISE` is required
for any unresolved mathematical identity, missing SPD/downdate assumption,
unhandled derivative term, hidden fallback, inadequate independent oracle, or
material testing gap.

## 6. Expected audit artifacts

Fable should return, under a unique versioned directory or a clearly cited
MathDevMCP artifact root:

```text
environment.json
source_hashes.txt
latex_lookup.json or .md
assumption_checks.json or .md
identity_checks.json or .md
math_to_code_checks.json or .md
generated_test_obligations.json or .md
review_packet.json
fable_audit_report.md
```

The report must state exact commands, elapsed times, tool/backend availability,
source digests, diagnostic-only limitations, and all abstentions. If a backend
is unavailable, Fable must record the limitation and continue with the bounded
human/numerical review rather than fabricating a successful proof result.

## 7. Nonclaims and handoff boundary

This memo does not authorize code edits, package/environment mutation,
MacroFinance changes, public release, NeuTra training, HMC/NUTS, or scientific
claims. A positive Fable verdict is a plan-review result only. Implementation
still requires the execution phases, focused tests, exact DZ5 rare-row gates,
execution-mode qualification, retained artifacts, and a separate terminal
result review.
