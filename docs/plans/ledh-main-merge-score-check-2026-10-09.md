# Local main integration and LEDH score evidence check, 2026-10-09

## Question and scope
Merge the completed LEDH changes from local main into the execution branch,
preserve existing work, and assess the saved likelihood/score comparisons. This
supports the original surrogate-HMC investigation; it cannot establish posterior
validity. Source: 285b43e111124b0f3337f81032905591b9a54324. Branch before merge: f5e69d716818ff9ecf28b732a8952f5ec7f87739. The named October 8
comparison exists byte-for-byte on local main and SQMC but is absent from fetched
origin/main. Local main is therefore the relevant source.

## Evidence contract and research intent
Mechanism: covariance-guided mixture proposal, marginal importance weights and
analytical recursive scores. Comparator: the saved old complete filter variant
on identical data and paired designs. The dirty worktree is not the old
scientific baseline. Inspect controls, references and paired uncertainty before
interpreting old/new differences. Multiple changed controls prevent attributing
gains to one code change.

Engineering pass: clean merge, recovery of prior tracked edits and staging,
preserved untracked files, and focused value/score, mixture/proposal and reporting
tests. Conflicts, missing data and failing invariants trigger repair and veto a
successful-integration claim until resolved.

Scientific criterion: saved paired uncertainty intervals for model/dataset-specific
likelihood and score errors. Invalid/nonfinite rows veto the affected comparison.
Losses to simple comparators veto universal promotion, not continuation of the
research. Raw means and run times are explanatory unless supported by the declared
uncertainty. Inspect Kalman, refined integration, bootstrap and Gaussian comparators
separately by model where available; missing heuristic results are not passes.
No default promotion, new tuning, HMC run, universal improvement or conclusion
about untested range-bearing scopes follows from this check.

## Default and assumption audit
| Choice | Provenance/status | Risk and early check |
|---|---|---|
| Local main | Owner's named file; checked Git trees | Pin commit and exact blob to avoid wrong source |
| Preserve dirty work | Existing owner/campaign edits | Save both patch layers; tracked-only stash; zero incoming untracked collisions |
| Saved FP64 GPU/XLA comparisons | Frozen diagnostic evidence | State precision/scope; no FP32-default transfer |
| CPU-only tests | Engineering/reference exception | Hide CUDA; no GPU performance claim |
| Old/new controls | Complete-variant comparison | Record confounding; no single-mechanism attribution |
| Paired t intervals | Existing fixed-data analysis | Few datasets and reference bias; no broad ranking |

## Execution, budget and artifacts
One pinned merge with normal conflict resolution. Use a tracked-only stash,
keeping all preexisting untracked evidence in place. Keep stash and patches through
verification. Do not change the SQMC worktree or push. Use existing tftwogpu Python;
hide CUDA before framework import and bound CPU threads. Save exact commands,
logs, times and outcomes. Budget: 30 minutes of focused engineering checks, at most
one localized repair/retry per distinct merge defect; no long scientific campaign.
Evidence root: docs/plans/artifacts/ledh-main-merge-20261009T053052Z.

## Skeptical audit and pre-mortem
Pass for bounded integration. The wrong-source issue was found: remote main lacks
the changes, while pinned local main contains them. Saved comparisons have the
actual old arm; a dirty worktree comparison would be misleading. FP64/T50 evidence
cannot establish FP32, all horizons/dimensions or HMC readiness. Changed controls
confound attribution. Broken integration, corrupted/missing artifacts, invalid
reference assumptions or exhausted budget stop the affected check. Candidate
underperformance does not reject the research direction.
