# NeuTra Self-Contained Monograph Expansion Plan

Date: 2026-07-17

Status: `COMPLETE`

## Objective

Expand the NeuTra chapter included by `docs/main.tex` into a self-contained
account of the method and the available BayesFilter evidence. A reader should
be able to derive, implement, and critically interpret the method without
consulting the original NeuTra paper.

## Skeptical Audit

The pre-execution audit found four material gaps in the existing chapter:

1. HMC mechanics and Metropolis correction were assumed rather than derived.
2. The BayesFilter dense-IAF map was named but its masks, triangular Jacobian,
   bounded log scale, stage composition, fixed affine lift, and exact log
   determinant were not written out.
3. The distinction between variational approximation error and the exact
   Jacobian-corrected HMC target was asserted but not proved.
4. The evidence table named models without enough equations, priors, parameter
   charts, data horizons, filter likelihoods, sampling budgets, or
   parameter-level results to interpret the claims independently.

The plan passes after revision because the chapter will derive the method in
project notation, identify the actual implementation route, reproduce the
common truth-tail results from preserved archives, and keep heterogeneous
historical screens visibly separate.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the monograph provide a self-contained, implementation-faithful, source-grounded account of BayesFilter NeuTra and its experimental evidence? |
| Baseline | Existing `docs/chapters/ch26b_neutra_transport_hmc.tex`, the locally archived NeuTra paper, BayesFilter implementation code, and preserved result artifacts. |
| Primary pass | All core equations are derived; the tested map matches the active code; every reported number is traceable to a preserved artifact; the full monograph builds. |
| Vetoes | Conflated publication metadata; generic IAF substituted for the tested implementation; warm-up pooled with posterior draws; folded R-hat represented as the only R-hat; historical screens represented as common truth-tail tests; unsupported model or result details; new scientific claims inferred from descriptive results. |
| Explanatory diagnostics | Chapter length, table layout, LaTeX warnings outside the chapter, and bibliography coverage. |
| Not concluded | No new NeuTra experiment, calibration theorem, sampler superiority result, filter-exactness claim, production readiness, or universal reliability. |
| Artifact | Expanded chapter, corrected bibliography, source/claim ledger, built `docs/main.pdf`, and terminal result note. |

## Default And Assumption Audit

| Choice | Provenance | Status and risk control |
| --- | --- | --- |
| Standard-normal latent reference | NeuTra method and BayesFilter training config | Method definition; derive the objective explicitly. |
| Three dense autoregressive stages with reversals | BayesFilter `PlainDenseIAFTrainingConfig` and frozen artifacts | Tested implementation, not a universal architecture default. |
| Reverse KL | NeuTra method and active code | Training objective only; downstream HMC remains the scientific gate. |
| Modern R-hat | Active BayesFilter diagnostics | Define as the per-parameter maximum of rank-normalized split and folded rank-normalized split R-hat. |
| Historical model count | Cross-repository evidence ledger | Count families and posterior-target configurations conservatively; do not count seeds or tuning arms. |
| Common truth-tail set | Hash-verified retained archives | Report five eligible configurations and all 38 inspected parameters. |

## Execution

1. Audit the primary NeuTra source, local bibliography, implementation, and
   result artifacts.
2. Expand background, HMC, transport, IAF, training, geometry, algorithm, and
   diagnostic derivations.
3. Add self-contained model and protocol descriptions plus parameter-level
   common truth-tail results.
4. Add full historical evidence descriptions with explicit protocol and claim
   boundaries.
5. Run an independent bounded review, focused source/number checks,
   `git diff --check`, and a full LaTeX build.
6. Write a terminal result and drift audit.

## Stop Conditions

Stop rather than claim completion if the tested transport cannot be reconciled
with the documented equations, a reported result lacks a preserved source, a
material mathematical review finding remains unresolved, or the chapter does
not build.

## Execution Close

Completed on 2026-07-17.  The chapter is included by `docs/main.tex`, the full
monograph builds, all 38 common parameter rows match the two structured result
archives within printed precision, and the final NeuTra log slice has no local
warning.  Claude's bounded mathematics, experimental-reporting, and added-proof
reviews all ended `VERDICT: AGREE`.  The terminal record is
`docs/plans/bayesfilter-neutra-self-contained-monograph-expansion-result-2026-07-17.md`.
