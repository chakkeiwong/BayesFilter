# Phase 7 Focused Check Log

Date: 2026-07-14

## Full Monograph Build

```text
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
exit: 0
Output written on main.pdf (387 pages, 1532934 bytes).
Latexmk: All targets (main.pdf) are up-to-date
```

Output hashes:

- `docs/main.pdf`:
  `aa72610b424a749ee98e93ca935cdd559388f507b0099ca92e4ffdb86bf38df6`;
- `docs/main.log`:
  `714a0f9cf45e76af26ae6fcef9dc0d2cace1da1e63b9828d80d9186572279558`.

The successful build retains four duplicate-label names and eleven unresolved
citation occurrences. The duplicate labels predate Phase 7 in `HEAD`, the
unresolved keys are absent from the `HEAD` bibliography, and Phase 7 introduced
no citation. These are nonfatal pre-existing warnings, not evidence that the
edited chapters failed to integrate.

## Static And Semantic Checks

```text
git diff --check -- \
  docs/chapters/ch32c_entropic_ot_sinkhorn.tex \
  docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex \
  docs/main.pdf
exit: 0
```

Targeted searches verified:

- each of the seven new labels occurs exactly once;
- fixed prepared ridge and residual design;
- explicit `Y=Q/M` plus quotient JVP/VJP;
- direct-plus-transport source and weight composition;
- one normalized-log-weight pullback;
- transition-first likelihood increment and active/inactive carry semantics;
- historical-only raw paths and no fallback; and
- explicit nonclaims for Kalman, nonlinear, admission, HMC, leaderboard,
  default, and release status.

The detailed equation-to-code crosswalk is in `code-anchor-audit.md`.

## Artifact Hashes

- entropic chapter:
  `5a368ba1e98aaeb1b2feba4658d50dc8523e080c255a4ca8ee889b588da30d50`;
- custom-gradient chapter:
  `06a5815a49f2b92d3037690e344a4179465844891aedc784dfefb88cb6315faa`.

## Interpretation

The two chapters now describe the checked canonical finite program and total
derivative consistently, and the full monograph builds. This is documentation
consistency evidence only. It does not establish numerical adequacy, Kalman
equivalence, nonlinear validity, v2 admission, HMC readiness, leaderboard
completeness, default readiness, release readiness, or integrity clearance.
