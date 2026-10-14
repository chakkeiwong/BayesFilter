# Source-trial launch result — September 25, 2026 HKT

The additional 48-hour budget is active. TG/IRG short source probes passed in
74.946067 /111.836848 seconds. IRG reached its intended gradient consumer.
Guarded MDPF smoke passed in87.684195 seconds with1185 finite-gradient checks;
original checkpoint handoff and paired epochs passed. All23 focused tests pass.

Attempt16 started at02:07:10 HKT September25 on4080SUPER, PID1346434, with
43200-second limit. Immutable source-repair11 and manifest-repair11 preserve
executed code, author hash, exact command, original400/800 caps, seeds and
environment. Pretraining completed137 epochs through source early stopping in
671.474183 seconds. Full filtering training is now active.

Follow-up PID1357219 waits without GPU use, then evaluates5000 fresh T150
trajectories with900-second GPU limit. Both NLL and RMSE measure the NLL-trained
checkpoint: no MSE refinement is implied. It preserves effective source seed0;
fresh data seed925231 differs from training925201.

## Decision table

| Decision | Primary status | Veto status | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue NLL source trial | Full training active | Smoke finite; guards active | Long-run numerics and early stopping | Finish source schedule and untouched evaluation | Learned quality or replication |
| Keep iAPF open | Eq15 fit unresolved | Amplitude escape invalidates tiny-loss success | Solver/floor provenance | Qualified reconstruction, non-harm checks | Rejection of research direction |
| Keep synthetic/port gaps open | Not completed | Missing source consumer/port proof | Appendix B, whole-NLL expectation, TF parity | Resolve separately | Reference is runtime default |

## Inference status

| Item | Status |
| --- | --- |
| Hard veto screen | Guarded smoke passed; active trial not terminal |
| Statistically supported ranking | None |
| Descriptive-only differences | Short runtimes and ongoing loss trajectory |
| Default readiness | Not established |
| Next evidence needed | Completed trials, conditional classical/heuristic checks, independent repeats/uncertainty, TF consumer parity |

Interim red-team: falling source loss could coexist with poor long-horizon
tracking. Untouched errors against classical baselines can overturn optimism.
Single-seed performance and source/paper differences remain the weakest evidence.
No unbiased-gradient or method-superiority conclusion is made.
