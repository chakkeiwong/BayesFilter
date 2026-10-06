# Active checkpoint

Task: commit/push completed NeuTra study, document/review synthetic results,
execute reviewed short-first q20 NAF training tests.

Plan and results: `bayesfilter-q20-naf-forward-reverse-{plan,results}-2026-10-06.md`.
Full monograph compiled; changed pages 573--576 inspected. Twelve random pairs
passed synthetic screens; finite tail residuals prohibit uniform-whitening claims.

Shared branch preserve/shared-main-before-fab-20260926, HEAD 70a6d7e96.
Isolated integration worktree `/tmp/BayesFilter-neutra-q20-20261006` on
integrate/neutra-q20-20261006, based on origin/main 11ebdfb92. Scoped source and
docs staged, no commit/push yet. Preserve shared unrelated HMC changes. Keep
remote precision APIs/docs, log-domain DSF underflow fix and q20 event-monitoring
section when syncing. Path list `/tmp/neutra-q20-integration-paths.txt`.

q20 uses four free parameters, T30 UKF target and explicit
`tensorflow_eigh_strict_factor_cached`; default custom-op backend stalled twice.
Complete pricing passed derivative checks but rejected prior importance teacher.
Eight saved SMC populations replayed all 800 values after exact chart Jacobian
correction; exploratory teacher screen passes. These represent two known sign
regions, not exhaustive discovery. Teacher output:
`artifacts/q20-naf-forward-reverse-2026-10-06/reuse-smc-20261006T111325`.

Fit `fit-20261006T111910` completed with finite 128/512 forward and 32 reverse
updates, zero clipping, coverage/reload checks passing, and full 1,000-point
endpoint probes. Held-out cross entropy fell 18.02 -> 6.53 before reverse;
reverse median score residual fell to 8.34, but p99 remained 479.51 and max
745.13. This is a viable short training call chain, not a correctly whitened
q20 map. The reviewed second seed at 2048/128 is now active; do not launch the
8192/16384 ladder before evaluating this calibration. The initial fit attempt failed before training due a diagnostic
control dtype and is preserved as a charged infrastructure attempt.

Local state.json and shared accountant preserve all charges. Remaining after
the completed fit: 109050.51 GPU-process and 101365.42 CPU-core seconds. The
initial pilot exposure cap remains 7200 GPU seconds; the short campaign used
2896.77 GPU-process seconds in total.

Active continuation: fit-extended-20261006T114852, PID 3531329, session 73849.

Next: monitor this bounded continuation, preserve results, synchronize final
reviewed files, validate integrated sources/manuscript, commit and push to main
without changing the shared checkout branch. Synthetic study should not be
rerun. A later long ladder requires a new reviewed continuation note.
