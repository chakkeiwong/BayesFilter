# NeuTra HMC Program Phase A Drift Matrix

Date: 2026-07-15  
Status: `CLOSED_AFTER_CLAUDE_AGREE`

| Audit question | Evidence checked | Verdict | Repair or residual limit |
| --- | --- | --- | --- |
| Active shared-core route | Route ledger, source discovery guard, migrated LGSSM wrapper, active sampler scan | pass | Local TFP HMC construction remains only in canonical `bayesfilter/inference/neutra_hmc.py`. |
| NumPy and host callbacks | Active inference/training source audit and route-policy tests | pass | No repository NumPy, `tf.numpy_function`, or `tf.py_function` in active claim-bearing paths. |
| Warm-up archive and exclusion | S1, F0, and F2 result JSON plus tensor sidecars | pass | Every serious sequential run archives warm-up separately and excludes it from posterior summaries. |
| Diagnostic roles | Core tests and serious results | pass | Warm-up uses recent-window modern R-hat; retained decisions use cumulative modern R-hat and full convergence callbacks. Modern R-hat is the maximum of rank-normalized split and folded rank-normalized split R-hat. |
| Caps and minimums | Core defaults/configs and serious results | pass | Warm-up and retained caps are 10,000 per chain; confirmation minima were 2,000 warm-up and 4,000 retained. |
| Health/status vetoes | S1/F0/F2 results | pass | No hard veto in an admitted kernel. F0 step 0.8 was rejected locally and was not allowed to veto distinct healthy candidates. |
| Seed separation | Controller tests and serious result manifests | pass | Warm-up/retained and admission/confirmation roots are distinct; S1 training seed is `(20260715,1203)`, F0 fixture seed is `(20260715,701)`, and F1 final seed is `(20260715,8201)`. |
| Target and comparator identity | F0 identity ledger, S1/F2 results, frozen payload hashes | pass after repair | Added a computed stable self-hash to `f0/fixture_identity.json`; original and new target signatures remain unequal and F2 binds the F0 comparator. |
| Training implementation | S1/F1 artifacts and training tests | pass | GPU/XLA, one compiled batched loop, memory growth, no screen-weight reuse, exact frozen parity, no scalar/sample-axis Python fallback. |
| Proxy promotion | Program, selection ledger, result inference tables | pass | Acceptance and loss nominate or explain only. Downstream HMC/comparator/recovery gates decide S1/F2. No recipe or sampler ranking is claimed. |
| Candidate versus direction failure | F0 repair record and all result notes | pass | One unhealthy F0 kernel was rejected without invalidating the target or later repair; no candidate failure was upgraded into universal rejection. |
| Artifact integrity | 48 embedded file hashes, 120 tensor hashes, 35 byte counts, and 75 stable self-hashes | pass after repair | All 168 file/tensor hashes and every byte count/self-hash verify. The sole missing self-hash was added to the F0 identity ledger and recomputed. |
| Required run manifests | Serious results and terminal manifest | pass with disclosed limitation | F1 had runtime/GPU facts but no consolidated command manifest. Added `phase-a/serious_run_manifest.json`; its command strings are explicitly reconstructed, not claimed as contemporaneous transcripts. |
| Phase status and handoff records | Master, subplans, and results | pass after repair | Stale ready states changed to completed; Phase A marked in progress pending review. |
| Duplicate C2 wording reported at crash handoff | Exact current master-plan scan | not reproduced | No duplicated C2 sentence exists in the current master; no speculative deletion performed. |
| Claim scope | S1/F2 inference tables and terminal contract | pass | Claim is limited to one additional training seed and one additional fixture in the same 18D LGSSM family. |

Local audit verdict: `PASS_AFTER_REPAIR`. No implementation, numerical,
statistical, target, comparator, or artifact-corruption veto was found. The
remaining action is one bounded Claude read-only review of the terminal packet.
