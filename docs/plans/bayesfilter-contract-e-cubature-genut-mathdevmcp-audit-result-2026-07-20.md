# Contract E Cubature and GenUT Derivation Audit

Date: 2026-07-20

## Scope

This note records a documentation and mathematical-audit pass for two
non-fused residual designs for the staged Contract E route:

1. the replicated \(2d\)-direction spherical-radial cubature design; and
2. the Ebeigbe et al. generalized unscented transformation (GenUT).

The fused OT/moment optimization route is explicitly not pursued. The existing
sequence remains:

\[
\text{flow and current increment}
\to\text{positive entropic OT}
\to\text{barycentric cloud}
\to\text{residual injection}
\to\text{Cholesky restoration}.
\]

The LGSSM is a controlled diagnostic harness for future high-dimensional
nonlinear filtering. This work does not claim that LGSSM estimation is the
scientific endpoint, and it does not implement or test NAWM.

No runtime source-code implementation was added by this pass.

## Evidence Contract

| Item | Status |
|---|---|
| Research question | What moment, likelihood, and score statements are actually valid for the two staged residual candidates? |
| Baseline | Canonical staged Contract E--Chol finite program |
| Primary criterion | The document must prove the finite design/restoration identities and state the exact boundary of the likelihood and score claims |
| Veto diagnostics | TeX failure, algebraic contradiction, negative-weight use as a positive OT marginal, or an unsupported exact-filtering claim |
| Explanatory diagnostics | MathDevMCP proof-audit status, symbolic simplification, source-paper section checks, and LaTeX warnings |
| Nonclaims | No exact filtering likelihood, exact filtering variance, exact posterior score, or production readiness is established |
| Artifact | docs/chapters/ch32c_entropic_ot_sinkhorn.tex, this note, and /tmp/bayesfilter-latex/main.pdf |

## Mathematical Changes

The chapter now states explicit dimensions, \(N=2dM\), population covariance
denominator \(N\), strict positive-definiteness assumptions for all Cholesky
arguments, and fixed-branch assumptions for the derivative claim.

The cubature section includes:

- the replicated design identities
  \[
  N^{-1}{\bf 1}^{\mathsf T}\Xi_c=0,\qquad
  N^{-1}\Xi_c^{\mathsf T}\Xi_c=I_d;
  \]
- an affine first/second-moment proposition;
- the realized covariance expansion
  \[
  \widetilde\Sigma=\Sigma_+
  +C_{+c}B^{\mathsf T}+BC_{+c}^{\mathsf T}+BB^{\mathsf T},
  \]
  showing why residual injection alone is not the restoration proof; and
- the Cholesky restoration identity
  \[
  \Sigma_{E,c}-\Sigma_w=\lambda(I-A_cA_c^{\mathsf T}).
  \]

The GenUT section includes:

- the axis construction and its four scalar moment equations;
- the explicit positivity distinction \(k_a>s_a^2\) versus \(w_0\ge0\);
- an exact equal-weight replication condition;
- a detailed affine mean/covariance proof; and
- the same staged value/variance/score boundary as cubature.

The shared proposition states:

- the current increment is unchanged because the reset is post-increment;
- both resets have mean \(\mu_w\) and the same ridged covariance identity when
  the realized covariance is used; and
- the score is the total derivative of the new finite scalar only when all
  dependencies are differentiated on a fixed branch.

## Ebeigbe Source Audit

Source:
docs/Generalized unscented transformation for forecasting non-Gaussian processes Ebeigbe(25).pdf

Classification: DIRECT_METHOD, local full-text source available, no
retraction or quarantine finding in this local audit.

Technical anchors inspected:

- Section III, Eqs. (22)--(32): multidimensional point/weight construction,
  diagonal whitened skewness/kurtosis constraints, and feasibility condition.
- Algorithm 1: unconstrained GenUT construction.
- Section IV, Theorem 1 and Eqs. (35)--(39): mean, covariance, and selected
  diagonal third/fourth moment identities.
- Section V, Algorithm 2, and the surrounding discussion: support constraints,
  possible negative weights, and loss of exact fourth-moment matching after
  constraint enforcement.
- Section VI and Tables II--III: selected nonlinear transformation
  experiments, treated as empirical examples rather than general filtering
  validity.

Allowed source-supported claim: GenUT matches mean and covariance and selected
diagonal third/fourth moments in its declared whitened coordinate system when
the unconstrained construction is feasible.

Forbidden source overclaim: GenUT does not establish exact filtering
likelihood, exact filtering score, full correlated third/fourth tensor
matching, or validity of the staged Contract E particle program.

Backward/forward snowballing: no network/API metadata lookup was used in this
pass. Citation counts, venue rankings, and forward-citation coverage are
therefore recorded as unavailable rather than inferred. The chapter uses the
paper only for the direct method and its stated limitations.

## Verification

### MathDevMCP

MathDevMCP was already installed at /home/chakwong/MathDevMCP; no install or
environment mutation was needed.

Doctor command:

    cd /home/chakwong/MathDevMCP
    PYTHONPATH=src python -m mathdevmcp.cli doctor

Result: passed. LaTeXML, Pandoc, Sage, Lean, LeanDojo, SymPy, and the MCP
runtime were available. The doctor report noted an unrelated magic-pdf
dependency conflict in the active environment; it did not affect these
bounded checks.

The whole repository was not used for the proof audit because unrelated legacy
LaTeX contains a parser-invalid brace-depth construct. The selected corpus was:

- ch32c_entropic_ot_sinkhorn.tex;
- preamble.tex.

Bounded audit-derivation-v2-label results:

| Label | Result | Diagnostic boundary |
|---|---|---|
| prop:bf-eot-cubature-design-moments | unverified | missing explicit conformability constraints in the extracted obligations |
| prop:bf-eot-cubature-affine-moments | unverified | one manual-formalization obligation and two ambiguous extracted rows |
| prop:bf-eot-cubature-restoration | unverified | missing matrix-shape constraints in extracted obligations |
| prop:bf-eot-genut-axis | unverified | manual formalization required for the radical/weight derivation |
| prop:bf-eot-genut-moments | unverified | manual formalization required for tensor/affine identities |
| prop:bf-eot-genut-positivity | inconclusive | parser did not emit a certifiable bounded equality |
| prop:bf-eot-both-residual-boundary | unverified | fixed-branch/matrix-shape obligations remain outside the bounded backend |

These are diagnostic abstentions, not mismatches and not proof
certifications. The chapter supplies the missing assumptions and manual
proofs; MathDevMCP does not certify the complete matrix/tensor derivation.

### SymPy

The following identities simplify exactly:

\[
\begin{aligned}
-b_au_a+c_av_a&=0,\\
b_au_a^2+c_av_a^2&=1,\\
-b_au_a^3+c_av_a^3&=s_a,\\
b_au_a^4+c_av_a^4&=k_a,\\
v_a-u_a-s_a&=0,\\
u_a^2+s_au_a+s_a^2-k_a&=0.
\end{aligned}
\]

### LaTeX

Command:

    cd docs
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=/tmp/bayesfilter-latex main.tex

Result: passed; PDF written to /tmp/bayesfilter-latex/main.pdf.

The build retains pre-existing duplicate-label, undefined-citation, and
overfull/underfull-box warnings. No fatal error was introduced by this pass.

## Decision and Nonclaims

| Decision | Primary criterion | Veto status | Uncertainty | Next action |
|---|---|---|---|---|
| Retain staged Contract E | Fused route is not needed for this derivation pass | no veto | implementation feasibility not tested | keep runtime route unchanged |
| Cubature residual is mathematically well-defined under stated assumptions | design and restoration identities derive in local notation | no algebraic mismatch found | MathDevMCP abstains on full matrix formalization | if implemented, run fixed-branch moment/value/score gates |
| GenUT is a separate candidate, not a positive OT marginal by default | source and positivity boundaries are explicit | negative \(w_0\) is a hard route veto for positive OT | support-constrained variant changes its moment target | implement only after positive representation is specified |
| Exact filtering value/variance/score | not established by either derivation | exactness claim vetoed | finite-particle and reset errors remain | compare against diagnostic truth only |

The current increment is exact relative to the executed finite program's
pre-reset ordering. The full future likelihood, reset variance relative to the
exact filter, and posterior score can change. A same-scalar total derivative is
not an exact-posterior score theorem.
