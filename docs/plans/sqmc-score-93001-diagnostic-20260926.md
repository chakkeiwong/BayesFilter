# Diagnose the first SQMC score component for seed 93001

Active checkout: /home/chakwong/BayesFilter-SQMC; branch sqmc-development.
User question: why do the three Halton/Hilbert variants have the wrong first-score sign?

## Evidence contract and skeptical audit
Reproduce the saved P44 D=3, T=2, N=12 float64 CPU diagnostic using each route's saved controls and exact saved seeds. These are the bounded repair configurations (two flow substeps), not production-performance evidence. The saved per-route tuning artifacts under ../plans/artifacts/sqmc-repair-20260925/06-tuning-workflow bind their original scope; new interventions are UNTUNED diagnostics and cannot be promoted.

First determine whether the analytical derivative of the executed finite likelihood agrees with a central finite difference at fixed data and random inputs. Compare the same scalar and inspect ancestry at both perturbations. Require replay agreement within 1e-9 and finite-difference agreement within 2e-6*(1+abs(score)) where ancestry is unchanged, with h=1e-4 and 1e-5. Failed parity or replay invalidates dependent interpretation and triggers debugging. A branch change is a localization diagnostic, not proof of a derivative defect. Preserve raw traces.

Then separate contributions from the first and second observations against matched exact Kalman prefix likelihood derivatives. An independent Gaussian integration over process and observation noise at the first observation isolates initial-cloud/ancestry error from process-draw/flow error. Shared inputs, posterior weights, ancestor identities, and empirical state-noise cross moments are explanatory diagnostics.

If necessary, perform bounded interventions: keep all saved random inputs while increasing flow steps; keep data fixed while changing only the random scramble or replacing initial/process inputs; increase N solely to diagnose finite-cloud sensitivity. Such interventions are not fresh holdouts, tuning, a convergence proof, an accuracy campaign, or evidence ranking methods. Do not equate parity with Kalman accuracy or shared-route errors with independent replication.

Skeptical audit passed for diagnosis: the exact target, theta coordinates, data, dtype, and computation are matched; data stay fixed during differentiation; derivative and filtering errors remain distinct; the three Hilbert routes share randomness; the cap ablation is not independent; original held-out datasets are now diagnostic. Avoid attributing the failure to small N or Hilbert sorting without a discriminating intervention.

## Defaults and assumptions
| Choice | Provenance and status | Risk | Early check |
|---|---|---|---|
| N=12, T=2, saved controls, theta | Original failed diagnostic; frozen reproduction baseline | Too small for general accuracy claims | Replay scalar and score exactly |
| Same data and Halton inputs under perturbation | Derivative definition | Randomness or data could silently change target | Save data, inputs, and ancestry |
| h=1e-4,1e-5 | Diagnostic hypotheses | Truncation, cancellation, rank changes | Compare both slopes and discrete indices |
| CPU float64, non-XLA | Explicit reference/debug exception matching saved run | Does not test production GPU/XLA | Record intentional GPU hiding |
| Fixed controls under interventions | Causal diagnosis, not transferred tuning | New scope is untuned | Label all interventions and forbid promotion |

No learned/optimized method is promoted or ranked. Exact Kalman, finite differences of the actual scalar, and conditionally integrated Gaussian predictions are the constructed mathematical comparators.

Budget: at most 900 seconds CPU numerical execution, three attempts, within the remaining original repair budget (360.097/3600 CPU seconds previously used). No GPU, package mutation, HMC, full campaign, or production default change. Save attempts in docs/plans/artifacts/sqmc-score-93001-20260926/attempt-NN. Stop on budget exhaustion, unresolved reproduction mismatch, nonfinite outputs, or a target change. Write a result note and checkpoint with exact next action.

## Completed diagnosis

Both attempts exited 0. Replay and fixed-ancestry finite differences passed.
The first observation and shared small Halton cloud dominate the error;
inverse-CDF duplication amplifies it. Larger-N and flow-resolution interventions
separate these explanations. No default or algorithm change was made.
See ../benchmarks/sqmc-score-93001-diagnosis-20260926.md for evidence, inference
limits, and exact commands. Measured post-import execution 82.038 s; timeout
bounds 600 s, within the 900 s allowance. No further attempt is justified by
the local question. Self-review found no unsupported ranking; systematic
approximation error outside the checked cells remains unexcluded.
