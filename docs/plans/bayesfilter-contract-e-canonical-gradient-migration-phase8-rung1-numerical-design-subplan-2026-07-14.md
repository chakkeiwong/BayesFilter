# Phase 8 Subplan: Target-Prefix Numerical Design

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_INSTRUMENTATION_PASSED_NUMERICAL_ARMS_OWNER_REQUIREMENTS_BLOCKED`

## Phase Objective

Define and instrument scientifically interpretable ridge and finite-transport
hypotheses for canonical LGSSM target-prefix work without choosing settings from
the completed transferred-fixture smoke. This phase first closes diagnostic
coverage and freezes a parameter-domain feasibility rule, raw-covariance bias
budget, and transport convergence rule. It does not execute a new Contract E
target value or gradient until a reviewed amendment makes those gates
executable.

## Entry Conditions

- Phases 0-7 and Rung 0A/0B are closed at their stated narrow gates.
- Preparation/telemetry infrastructure is closed.
- The CPU-hidden `T=1,N=4` transferred-fixture smoke passed wiring only. Its
  numerical differences and telemetry magnitudes are explanatory and cannot
  nominate settings.
- Formal Phase 1 same-callable FD remains inconclusive because callable-specific
  endpoint-value and score absolute-error bounds are absent.
- No `T=10`, GPU, or `T=50,N=10000` candidate result may be observed here.
- The owner primary statistical amendment is still required before any
  primary-shape pilot or audit.

## Research Intent Ledger

| Field | Contract |
| --- | --- |
| Main question | What diagnostics and pre-result requirements are needed to define a fixed Contract E ridge and finite-transport schedule that are valid over the declared target parameter domain? |
| Mechanisms | Ridge controls positivity of `gap + lambda I` but creates exact raw covariance residual `lambda(I-AA^T)`; finite Sinkhorn controls marginal error independently of row-quotient validity |
| Promotion criterion | None in the instrumentation slice; a later arm requires reviewed executable feasibility/bias/convergence rules frozen before output |
| Promotion veto | Any selection from the smoke output; center-dependent stopped ridge; result-dependent ladder expansion; hidden raw fallback |
| Continuation veto | Inability to define a parameter domain or downstream bias/error budget; missing full marginal diagnostics; corrupted preparation identity; campaign-clock exhaustion |
| Repair trigger | Missing telemetry, serialization, or a localized harness defect under unchanged mathematics |
| Explanatory diagnostics | gap eigenvalues, required positivity ridge, raw residual identity, factor condition proxies, row/column marginal residuals by fixed iteration schedule |
| Must not be concluded | target numerical adequacy, Kalman equivalence, FD correctness, GPU/HMC/default/leaderboard readiness |

## Claimed And Computed Quantities

For a fixed prepared input and parameter point, the reset gap is

```text
G(theta) = Sigma_w(theta) - Sigma_plus(theta).
```

The first Cholesky factor exists only if

```text
lambda > -lambda_min(G(theta))
```

with an additional declared numerical safety requirement. A ridge computed from
the evaluated center and then stopped is wrong relative to the canonical total
gradient. Therefore a canonical ridge must be chosen outside the callable and
must satisfy the positivity rule over a declared parameter domain, not merely at
one observed center. The phase must report `G`, its symmetric eigenvalues, and
the required positivity lower bound, but those values cannot themselves select
the canonical ridge.

For any fixed ridge, the exact raw covariance change is

```text
mu_w       = sum_i w_i X_i,       Sigma_w = sum_i w_i (X_i-mu_w)(X_i-mu_w)^T
mu_plus    = N^-1 sum_i Y_i,      Sigma_plus = N^-1 sum_i (Y_i-mu_plus)(Y_i-mu_plus)^T
G          = sym(Sigma_w - Sigma_plus)
L_G        = chol(G + lambda I)
Y_tilde    = Y + Xi L_G^T
L_w        = chol(Sigma_w + lambda I)
L_tilde    = chol(Sigma_tilde + lambda I)
A          = L_w L_tilde^-1
X_star     = mu_w + (Y_tilde - mu_tilde) A^T
R_raw      = Sigma(X_star) - Sigma_w
           = lambda * (I - A A^T).
```

All covariances above use the canonical population convention `1/N` for
uniform clouds and the normalized probability weights `w_i` for the source
cloud. `A` is the row-vector affine map exactly as computed by the two
triangular solves in `ledh_contract_e_reset_tf.py`; no explicit inverse is
formed. The sign convention is output-minus-target, so the measured raw
residual is `output_covariance - target_covariance`. The ridged identity is
`A (Sigma_tilde + lambda I) A^T = Sigma_w + lambda I`.

The raw-bias gate must state a downstream norm and scale before candidate
output. A numerical-roundoff multiplier cannot justify this scientific bias
budget.

For finite transport, row quotient makes `Y=Q/M` well-defined whenever every
mass is finite and positive; it does not make a large marginal residual
adequate. The exact coupling convention is the one in
`annealed_transport_tf._filterflow_streaming_transport_from_potentials`: each
row target is the scalar `1`, each column target is `N*w_j` where `w` is the
normalized source probability vector, and the transport block entries are
nonnegative. For executed coupling `P`, record the full row mass
`m_row[i]=sum_j P[i,j]`, full column mass `m_col[j]=sum_i P[i,j]`, signed
residuals
`r_row=m_row-1` and `r_col=m_col-N*w`, and their maximum absolute norms
`max_i|r_row[i]|` and `max_j|r_col[j]|`. Report the corresponding relative
scales `max(1,max_i|1|)=1` and `max(1,max_j|N*w_j|)`; no post-result scale is
chosen. Each ladder point reports the final coupling produced by its own fixed
finite Sinkhorn schedule, not an intermediate iterate. A dense tiny reference
must reproduce these definitions exactly. Chunk comparisons use identical
finite settings and prepared inputs and remain engineering diagnostics until a
downstream error budget is declared.

## Required Instrumentation

1. Extend canonical telemetry with the already-computed pre-ridge reset
   quantities `plus_mean`, `plus_cov`, and `gap`, plus symmetric gap
   eigenvalues. This is reporting only and must preserve the scalar and score
   source-bound certificates exactly.
2. Extend the streaming diagnostic output with full column mass, target column
   mass, and their residual without constructing or retaining an `N x N`
   production matrix. Reuse the existing streaming block loops or add a bounded
   reporting pass; do not change the coupling, quotient, or derivative. The
   reporting pass may retain only `O(B*N*d + B*N)` state; a source/AST or graph
   audit must reject any `tf.Tensor` with both dynamic particle axes or any
   `N*N` materialization in the production diagnostic path.
3. Add schema/shape/finiteness tests and a tiny dense reference comparison for
   row and column residuals.
4. Re-run the frozen Phase 5 value/score identity and exact-JVP certificates to
   prove telemetry-only changes do not alter the canonical computation.

No target-prefix value or gradient is executed while completing items 1-4.

## Required Pre-Result Amendment

Before any numerical arm executes, the amended plan must freeze:

- the physical or HMC-coordinate parameter domain over which one prepared
  ridge must remain valid;
- how domain coverage is established without stopped candidate-dependent ridge
  selection, including any deterministic grid or analytical bound;
- a positive ridge candidate set derived independently of the completed smoke;
- the Cholesky safety rule and its downstream error-budget justification;
- the raw covariance residual norm, scale, and maximum permitted bias, with a
  scientific/downstream justification;
- epsilon and scaling hypotheses with provenance;
- a fixed Sinkhorn-step ladder and row/column convergence rule;
- chunk sizes and a chunk-drift error budget;
- exact dataset/estimator seeds, rung shape, attempt/compute cap, output roots,
  deterministic selection/no-selection rule, and tie rule;
- whether a candidate failure blocks promotion only or fires a true
  continuation veto.

The amendment may define a diagnostic ladder, but no observed magnitude from
the transferred smoke may determine its candidates or thresholds. If no
scientific downstream bias/error budget can be supplied, stop with
`BLOCKED_TARGET_NUMERICAL_REQUIREMENT` rather than inventing one.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can full ridge-feasibility and transport-convergence diagnostics be emitted without changing the canonical finite program, and can target numerical hypotheses be frozen independently of observed target output? |
| Baseline | Phase 5 v2 scalar/score hashes; Phase 1 gap/ridged/raw identities; exact dense tiny transport residuals |
| Primary criterion | Instrumentation is exact/source-bound/scalar-preserving; amendment states executable domain, bias, conditioning, convergence, selection, and budget rules |
| Hard vetoes | scalar/score drift; dense production allocation; raw route import; adaptive/stopped ridge; smoke-derived setting or threshold; incomplete marginal diagnostics |
| Explanatory only | all gap spectra, required-ridge lower bounds, condition proxies, and marginal-residual curves before an executable requirement is frozen |
| Not concluded | any numerical candidate pass, Kalman equivalence, formal FD, GPU/HMC/default/leaderboard readiness |

The preservation baseline is executable and exact: the float64 CPU-XLA Phase 5
v2 certificate is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase5/cpu-xla-same-callable-certificate-v2.json`
with SHA-256
`20ec133bc5aee47f5daf3dc54d4c3593189202b1c305640cbd30b5e33b4ca709`; its
center objective hex is `-0x1.55564a66d9848p+2`, score hex values are the five
entries recorded in that JSON, and center branch hash is
`bf25ece12ff85525620fdc1284abab76a35a54c28a4f998b89bbabd56aa005d7`. A fresh
certificate must reproduce those exact fields after telemetry-only edits.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Fixed parameter-independent ridge | Contract E policy | center-valid ridge fails elsewhere or creates material raw bias | domain gap spectrum plus raw residual identity | Required; magnitude open |
| Parameter domain | HMC use requires more than one center | arbitrary domain makes feasibility meaningless | owner/use-case declaration | Human/scientific blocker |
| Raw-bias budget | downstream likelihood/gradient use | convenient tolerance hides changed reset | explicit norm/scale and sensitivity argument | Scientific blocker |
| Sinkhorn schedule | finite target identity | too few iterations leave material marginal error | full row/column residual ladder | Hypothesis; rule open |
| Chunk sizes | production streaming implementation | accumulation drift or unfair comparator | identical-input tiling comparison | Engineering hypothesis |
| Formal FD | Phase 1 frozen design | heuristic screen without absolute bounds misclassifies derivative | separate callable error-bound plan | Separate blocker |

## Skeptical Plan Audit

Decision: `PASS_FOR_INSTRUMENTATION_ONLY; NUMERICAL_ARM_EXECUTION_BLOCKED`.

- The transferred smoke is not a baseline for selection and provides no
  threshold.
- Ridge feasibility and raw covariance bias are distinct; passing Cholesky does
  not pass covariance adequacy.
- Row quotient validity and Sinkhorn marginal convergence are distinct; a
  positive mass does not pass transport adequacy.
- The plan does not silently define a target parameter domain, bias budget, or
  convergence tolerance.
- Telemetry preservation is answerable without observing another target value
  or gradient.

## Required Checks And Reviews

1. Obtain one bounded review of this plan before instrumentation edits.
2. Implement telemetry only and add dense tiny reference/schema tests.
3. Run CPU-hidden focused checks, compilation, diff validation, and fresh
   source-bound Phase 5 certificates.
4. Write an instrumentation close/blocker record.
5. Draft the pre-result numerical amendment and obtain owner direction for any
   unresolved parameter-domain or scientific bias/error budget.
6. Review the amended execution design before running any numerical arm.

The instrumentation close must include:

- `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/rung1-numerical-design-instrumentation-attempt1/result.json`;
- a run manifest, focused-check record, and source/graph O(N) allocation audit
  in the same directory;
- `300` seconds per attempt, one planned attempt, and at most one localized
  harness retry in `attempt2/` with unchanged inputs; and
- the exact CPU-hidden command prefix
  `CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 MPLCONFIGDIR=/tmp`.

The instrumentation result is not a numerical arm and must contain no new
Contract E target value or score. A scalar/score drift or dense-state finding is
a hard blocker.

## Forbidden Claims And Actions

- Do not run or inspect another target-prefix Contract E value or gradient in
  the instrumentation slice.
- Do not choose ridge, epsilon, scaling, step count, chunks, or a threshold from
  the completed smoke.
- Do not use `0.05*sqrt(p)`, `0.1%`, or actual-SV `6%` as ridge/transport
  adequacy criteria.
- Do not use an evaluated center to create a stopped canonical ridge.
- Do not run `T=10`, GPU, primary shape, HMC, nonlinear migration, leaderboard,
  release, or integrity work.

## Exact Handoff Conditions

A target numerical arm may begin only after instrumentation preserves the
frozen scalar/score identities and a reviewed amendment freezes the parameter
domain, fixed ridge candidates, raw-bias requirement, conditioning rule,
transport convergence rule, chunk budget, seeds, shape, attempts, outputs, and
deterministic selection/no-selection behavior. Formal FD remains a separate
required gate before Phase 8 promotion.

## Stop Conditions

Stop for scalar/score drift, dense production state, missing full marginal
diagnostics, adaptive/stopped ridge semantics, inability to define the parameter
domain or scientific bias/error budget, campaign-clock exhaustion, or a
material review finding. Local telemetry/schema defects are repair triggers.
