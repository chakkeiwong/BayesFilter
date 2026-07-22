# Reset Memo: LGSSM Value and `q_scale` Bias Handoff

## Date
2026-07-20

## Context
We are reviewing the persistent disagreement between the canonical LEDH Contract E--Chol finite program and the exact differentiated Kalman LGSSM target.

The scientific state remains unresolved: the evidence supports a persistent finite-program bias, but not a mechanism.

MathDevMCP was initially blocked by a LaTeX parser failure, but that blocker is now understood and repaired.

## Decision / policy
Use this note as the handoff for the next agent:

1. Treat the current result as a bias-disagreement diagnosis, not a proof of the cause.
2. Do not interpret the MathDevMCP failure as a scientific verdict.
3. MathDevMCP repo-root search is now usable again; if a new parse failure appears, inspect visible `.tex` sources first.
4. Keep the next diagnostic time-local and same-scalar.

## What changed
- MathDevMCP `doctor` succeeded.
  - `sympy==1.14.0` is available.
  - `mcp==1.27.0` is available.
  - `lean-explore` is missing in the active Python.
  - `lean` version probing timed out.
- MathDevMCP `search-latex` on the repo root initially failed in the equation indexer with `ValueError: invalid brace depth in LaTeX display environment`.
- The verified trigger was the visible teaching notes in `docs/plans/` that redefine `\[` / `\]` as `\begin{equation}` / `\end{equation}`.
- The parser now masks command-definition lines before environment tokenization, so those macro redefinitions no longer create fake display-environment boundaries.
- A hidden `.claude/worktrees/...` copy reproduced the same issue, but it was only a copy of the same source problem, not the root cause.
- The review packet’s earlier hidden-file hypothesis is stale.

## Bugs / blockers resolved
- Symptom: MathDevMCP could not index the full LaTeX tree.
- Root cause: the equation locator was mis-parsing display-token text inside `\renewcommand{\[}{...}` and `\renewcommand{\]}{...}` lines.
- Resolution: command-definition masking in `mathdevmcp.equation_locator` and regression coverage in `tests/test_latex_index.py`.

## Verification already run
```bash
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli doctor
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli search-latex "bf-ledh-contract-e-total-source-pullback" --root /home/chakwong/BayesFilter --limit 3
pytest /home/chakwong/MathDevMCP/tests/test_latex_index.py -q
```

Observed:
- `doctor` returned a valid capability report.
- `search-latex` now returns results from `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` instead of crashing.
- Focused LaTeX index tests passed.

## Current policy
- Current evidence still points first at finite-particle / log-normalizer bias, Contract E reset amplification, and under-tuned geometry controls.
- The local lower-rung derivative evidence argues against a gross missing-term bug, but it does not certify the full Kalman target.
- Do not transfer `sinkhorn_steps` / `balance_steps` across scopes as a universal fix.

## Known limitations / cautions
- This is not a proof that the code is wrong in one specific line.
- This is not a proof that the review packet should be revised for mathematics.
- The next smallest discriminating artifact is a same-observation, same-stream time-local decomposition of value and `q_scale` score contributions.

## Suggested next steps
1. Run the time-local decomposition on identical observations and random streams.
2. Separate stationary, transition/proposal, observation-normalization, carried-weight, transport, and Contract E reset score contributions.
3. If another LaTeX parse issue appears, inspect the visible source tree before hidden worktrees.
