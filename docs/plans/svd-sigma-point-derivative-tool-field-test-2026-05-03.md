# Field test: MathDevMCP and ResearchAssistant for SVD sigma-point derivatives

## Date

2026-05-03

## Purpose

This note records a field test of the current MathDevMCP and
ResearchAssistant tooling for the specific problem that matters for
BayesFilter: source-backed derivation and audit of analytical gradients and
Hessians for an SVD sigma-point filtering likelihood.

The goal was not to finish the SVD sigma-point Hessian.  The goal was to test
whether the current tools are useful enough to reduce derivation risk, and to
identify which tool improvements are required before we rely on them for
industrial HMC filtering work.

## Cleanup action

Deleted stale guide:

- `/home/chakwong/MathDevMCP/docs/kalman-hessian-agent-guide.md`

Reason:

- It described an older, narrower MathDevMCP interface.
- The current interface has stronger surfaces: typed obligations,
  `audit-derivation-v2-label`, AST Kalman recursion auditing, parser policy,
  release evidence, and MCP-facing tool contracts.
- A search after deletion found no active current-doc references.  One
  historical reset-memo reference remains in MathDevMCP as provenance only.

## Tool health checked

### MathDevMCP

Command:

```bash
/home/chakwong/MathDevMCP/scripts/benchmark_gate_smoke.sh /home/chakwong/MathDevMCP
```

Result:

- Passed: 41/41 benchmark cases.
- Failed count: 0.
- Policy: all benchmarks must pass.

Command:

```bash
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli doctor
```

Result:

- `latexml`, `pandoc`, `sage`, `sympy`, and LeanDojo Python import are
  available.
- The direct `lean` executable version check failed because the toolchain tried
  to download during the check.
- This is not a blocker for the present field test because the field test used
  retrieval, parser policy, typed obligations, and diagnostic abstention rather
  than Lean certificates.

### ResearchAssistant

Command:

```bash
PYTHONPATH=/home/chakwong/research-assistant/src python -m research_assistant.cli \
  --root /home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph doctor
```

Result:

- Status: ok.
- Offline/provider-disabled mode.
- PDF text and structured-source inspection workflows are available.

The earlier path spelling `local_mcp_h1_esternal_agent_instructions.md` is no
longer current.  The present file is:

- `/home/chakwong/research-assistant/docs/validation/local_mcp_h1_external_agent_instructions.md`

## Scratch derivation fixture

A temporary LaTeX fixture was created at:

- `/tmp/bayesfilter_svd_field_test/svd_sigma_point_derivative_field_test.tex`

It contains a deliberately small one-step sigma-point Gaussian likelihood
derivative fixture with labels:

- `eq:svd-spf-innovation-cov`
- `eq:svd-spf-ll`
- `eq:svd-spf-score-differential`
- `eq:svd-spf-alpha-derivative`
- `eq:svd-spf-hessian-mixed`

The fixture records:

- innovation covariance `S = P_zz`;
- innovation `v = y - zbar`;
- solve variable `alpha = S^{-1} v`;
- first differential of the one-step Gaussian likelihood;
- derivative of `alpha`;
- a compact mixed Hessian formula.

This fixture is intentionally scratch material.  It is not a BayesFilter
source chapter and it is not evidence that the formula is correct.

## ResearchAssistant source support

ResearchAssistant was useful for source-backed lookup of the exact Matrix
Backprop LaTeX source already fetched into the BayesFilter research workspace.

Commands:

```bash
PYTHONPATH=/home/chakwong/research-assistant/src python -m research_assistant.cli \
  --root /home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph \
  source-labels --paper-id ionescu2015_matrix_backprop
```

```bash
PYTHONPATH=/home/chakwong/research-assistant/src python -m research_assistant.cli \
  --root /home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph \
  source-theorem --paper-id ionescu2015_matrix_backprop --label prop:svd
```

```bash
PYTHONPATH=/home/chakwong/research-assistant/src python -m research_assistant.cli \
  --root /home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph \
  source-equation --paper-id ionescu2015_matrix_backprop --label svd_K
```

Useful result:

- The source labels include `prop:svd`, `eqn:svd_dS`, `eqn:svd_dV`,
  `eqn:dLdX`, and `svd_K`.
- The `svd_K` source equation exposes the spectral-gap denominator
  `1 / (sigma_i^2 - sigma_j^2)`.

Interpretation:

- ResearchAssistant is useful as a source lookup and provenance tool.
- It helps prevent agents from citing vague memory of SVD/eigen
  differentiation.
- It does not prove the BayesFilter SVD sigma-point gradient or Hessian.

## MathDevMCP field-test results

### What worked well

MathDevMCP `search-latex` and `extract-latex-neighborhood` worked well on the
scratch fixture:

- All five labels were found.
- Label line provenance was correct.
- Local neighborhoods around the score, alpha derivative, and mixed Hessian
  were extracted cleanly.
- `audit-derivation-v2-label` preserved parser-policy evidence and returned a
  conservative status rather than over-certifying.

The v2 audit of `eq:svd-spf-hessian-mixed` returned:

- status: `unverified`;
- counts: total obligations 5, verified 0, mismatch 0, unverified 3,
  inconclusive 2;
- high-priority actions:
  - state or verify invertibility/SPD assumptions for inverse/solve operands;
  - state or verify conformable products;
  - state or verify square trace operands;
  - split ambiguous derivation rows into safer obligations;
  - perform solve-residual and conditioning diagnostics.

The v2 audit of `eq:svd-spf-score-differential` similarly returned
`unverified`, with actions for trace shape, invertibility, conformability, and
row splitting.

This is good behavior.  For this problem, a tool that abstains loudly is more
valuable than a tool that emits confident algebraic prose.

### What did not work well enough

`typed-obligation-label` did not extract the full right-hand side of the
multi-line equations.  For the mixed Hessian, it effectively focused on the
line containing `ell_ab` and did not build a rich obligation for the full
trace expression.  The optional context text was not enough to make the audit
aware of all declared assumptions as formal typed constraints.

`derive-label-step` is not yet reliable for matrix differential identities.  A
test of the standard inverse-solve differential pattern,

```text
d(S^{-1} v) = S^{-1}(dv - dS S^{-1}v)
```

returned `mismatch` because the current symbol extraction treated matrix
differential products too syntactically.  It also returned `mismatch` for the
literal alpha derivative expression in its labeled context, even though
`supported_by_context` was true.  This should be treated as a parser and
semantic-equivalence limitation, not as mathematical refutation.

## Verdict

The current tools are useful for derivation work, but only in a bounded role:

- ResearchAssistant is useful for primary-source discovery, exact source-label
  lookup, and claim provenance.
- MathDevMCP is useful for LaTeX retrieval, label provenance, implementation
  grounding, assumption prompts, parser evidence, and conservative abstention.
- Neither tool is currently strong enough to certify the analytical gradient
  or Hessian of an SVD sigma-point filter.

The correct operating mode is:

1. Use ResearchAssistant to fetch and inspect primary sources.
2. Write BayesFilter derivations in small labeled steps.
3. Use MathDevMCP to retrieve labels, audit obligations, and force missing
   assumptions into the open.
4. Treat `unverified` as a blocker for production claims.
5. Treat `mismatch` in matrix differential checks as a signal to inspect both
   the math and the parser.
6. Require independent numerical checks: finite differences, JVP/VJP parity,
   eager/compiled parity, and stress tests near spectral degeneracy.

## Required tool improvements

Before using these tools as the main audit infrastructure for SVD sigma-point
gradient/Hessian certification, improve:

1. Matrix differential tokenization:
   - preserve `dS`, `dv`, `dalpha`, `S^{-1}`, transposes, products, and trace
     operators without merging symbols such as `dS S^{-1}` into spurious
     tokens.
2. Multi-line equation obligation extraction:
   - extract full LHS/RHS pairs from `equation`, `align`, and split rows;
   - preserve complete trace expressions rather than only the first
     derivation row.
3. Formal assumption ingestion:
   - turn context text such as SPD, symmetry, vector/matrix shapes, and
     conformability into typed constraints used by v2 audit.
4. Shape-aware matrix calculus:
   - scalar/vector/matrix tags;
   - SPD and symmetry tags;
   - trace-square checks;
   - solve/inverse residual checks;
   - Hessian symmetry checks.
5. Matrix calculus proof templates:
   - `d log det S`;
   - `d(S^{-1})`;
   - `d(S^{-1}v)`;
   - `d(v^T S^{-1}v)`;
   - trace cyclic permutations under shape constraints;
   - product-rule expansions for mixed Hessian terms.
6. Spectral derivative templates:
   - source-backed SVD/eigen derivative obligations from Matrix Backprop;
   - explicit eigen/singular-gap denominators;
   - repeated or clustered spectral-value warnings.
7. Numeric diagnostic generation:
   - generate small SPD test cases;
   - compare analytic gradient and Hessian with finite differences and
     autodiff;
   - stress close eigen/singular values;
   - report condition numbers and solve residuals.
8. Code/derivation traceability:
   - map each labeled derivation step to code terms;
   - prefer solve-form implementation over explicit inverse;
   - require tests before an implementation claim becomes production-ready.

## Implication for BayesFilter SVD work

For SVD sigma-point filters, the next derivation pass should not try to produce
one monolithic Hessian formula.  It should create a labeled derivation ladder:

1. one-step Gaussian likelihood differential;
2. solve variable differential;
3. innovation covariance derivative;
4. sigma-point mean/covariance derivative;
5. square-root/SVD factor derivative or an explicit policy avoiding raw
   spectral tape gradients;
6. mixed Hessian terms;
7. shape, SPD, symmetry, and spectral-gap assumptions;
8. finite-difference and autodiff parity tests.

Only after this ladder survives source review, MathDevMCP audit, and numerical
parity should BayesFilter implement an SVD sigma-point custom gradient or
Hessian backend.

## Answer to the skepticism question

The skepticism is justified.

AI tokens are good at assembling plausible derivations and remembering common
identities, but this is exactly why they are dangerous for Hessians.  In SVD
sigma-point filtering, a missing transpose, a dropped product-rule term, an
unstated SPD assumption, or a hidden singular-value crossing can produce a
gradient that looks convincing but is not the derivative of the HMC target.

The right use of AI here is as a disciplined assistant:

- retrieve exact source material;
- keep notation and provenance organized;
- list proof obligations;
- look for missing assumptions;
- generate tests;
- act as an adversarial reviewer.

It should not be treated as the mathematical authority for a production HMC
gradient or Hessian.
