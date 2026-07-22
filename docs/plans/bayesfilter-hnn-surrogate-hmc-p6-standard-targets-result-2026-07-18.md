# P6 Result: Standard-Target Requalification

Decision: `P6_CLOSED_THREE_CELL_LOCAL_REQUALIFICATION_BLOCKERS`.

Funnel, ill-conditioned Gaussian, and German logistic regression were not run
under the corrected kernel because their exact historically selected NeuTra
charts cannot be reconstructed. This is the predeclared honest P6 outcome
`REQUALIFICATION_BLOCKED`, not a corrected-kernel failure and not evidence
against the historical NeuTra results.

## Source And Identity Audit

| Cell | Exact target/data available | Exact selected chart available | P6 decision |
|---|---|---|---|
| FUNNEL, paper-scale dimension 100 | yes | no: selected dense-IAF JSON under `/tmp` is gone | `REQUALIFICATION_BLOCKED` |
| ILLGAUSS, paper-scale dimension 100 | yes | no: selected dense-IAF JSON under `/tmp` is gone | `REQUALIFICATION_BLOCKED` |
| GERMAN-LR, gamma-scales dimension 51 | yes, including exact data and posterior reference | no: output is gone and historical runner did not serialize weights | `REQUALIFICATION_BLOCKED` |

The source repository is `/home/chakwong/python` at inspected commit
`b5193b2c734d10946ddaf364c737ca2311826028`. The German data SHA-256 is
`2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8`.
The complete machine-readable audit is
`docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p6/source_requalification_audit.json`.

For the synthetic cells, the historical paper-suite runner supports complete
transport-state serialization and reconstruction, but the result documents
place the decisive artifacts in `/tmp/neutra_gate1_track_a_dim100_iaf/`; those
files no longer exist anywhere found under `/home/chakwong`. For German, the
historical runner recorded architecture and training summaries but not the
trained IAF weights. Retraining any of the three would produce a new chart.
That can be a future fresh experiment, but it cannot answer this phase's exact
requalification question.

No GPU command was launched and no phase GPU budget was consumed. A
conservative `0.25` CPU wall-hours was charged for the source and filesystem
audit.

## Review And Decision

The skeptical audit found the original P6 subplan under-specified and patched
it to require exact target and chart reconstruction before training. Claude
then returned `VERDICT: AGREE` on the one-path refreshed subplan, specifically
confirming that the gates prevent requalifying a different target or chart.
The gates subsequently worked as intended.

| Decision field | Status |
|---|---|
| Primary criterion | not evaluated; exact chart precondition failed in every cell |
| Hard veto | cell-local identity/reproducibility continuation veto |
| Main uncertainty | fresh retraining may yield viable new charts, but was outside P6's requalification claim |
| Next justified action | proceed to independent P7 source/chart audit |
| Not concluded | corrected-kernel failure, historical-result invalidity, or cross-target impossibility |

There is no stochastic ranking to report. The strongest repair is to rerun a
separately planned fresh chart-training campaign with durable versioned
checkpoints. Doing so would change the P6 scientific question and is therefore
not smuggled into this program.

