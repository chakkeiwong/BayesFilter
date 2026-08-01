# P7 Result: NK And Rotemberg Requalification

Decision: `P7_CLOSED_FIVE_CELL_LOCAL_REQUALIFICATION_BLOCKERS`.

None of the five DSGE configurations entered corrected-kernel execution. The
historical result notes and portions of several target/preflight campaigns
survive, but no cell retains the complete selected NeuTra chart state required
to reproduce its transform and log-Jacobian. Under the reviewed P7 boundary,
fresh chart retraining would be a new experiment, not requalification.

## Cell Decisions

| Cell | Preserved evidence | Missing binding object | Decision |
|---|---|---|---|
| NK-ANALYTIC | result note and exact historical specification | documented selected chart root | `REQUALIFICATION_BLOCKED` |
| NK-REAL | result note and target/preflight history | selected parent root and resume training state | `REQUALIFICATION_BLOCKED` |
| NK-SVD-UKF | promotion note and exact selected candidate specification | selected diagonal chart and candidate artifact | `REQUALIFICATION_BLOCKED` |
| ROT-KF | result note, target preflight, launch summary | named selected replay/training state | `REQUALIFICATION_BLOCKED` |
| ROT-SVD2 | result note, target preflight, launch summary, parameter statistics | named final, replay, and training state | `REQUALIFICATION_BLOCKED` |

The complete audit is
`docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p7/source_requalification_audit.json`.
It records the inspected `/home/chakwong/python` commit, result hashes, exact
expected paths, and per-cell blocker. Launch summaries that merely name a
replay-state output do not reconstruct missing transport tensors. No summary
was treated as a checkpoint.

No GPU command was launched. P7 charged a conservative `0.5` CPU wall-hours
for the cross-repository source, artifact, and identity audit.

## Review And Interpretation

The skeptical audit patched P7 to separate fresh residual-force training
inside a replayed historical chart from forbidden chart retraining or
substitution. Claude returned `VERDICT: AGREE` on the bounded one-path review.
The resulting gate then classified every cell locally without blocking Tier A
or terminal synthesis.

| Decision field | Status |
|---|---|
| Primary corrected-kernel criterion | not evaluated in P7 |
| Continuation veto | selected chart state unavailable in each cell |
| Shared kernel/target validity | not invalidated |
| Main uncertainty | a future fresh-chart campaign may succeed but answers a different question |
| Next justified action | P8 synthesis of five completed Tier A cells and eight blocked Tier B cells |
| Not concluded | DSGE failure, historical NeuTra invalidity, or universal chart non-portability |

There is no stochastic ranking or corrected-kernel performance result in P7.
The negative result is an artifact-preservation and knowledge-transfer failure,
not evidence against the corrected neural-force method.

