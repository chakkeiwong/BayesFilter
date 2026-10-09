# q20 short-test checkpoint

User task: commit/push the NeuTra study, update and review the monograph, and
execute a reviewed q20 test with short whitening diagnostics before long training.
The bounded initial q20 plan is complete; no worker is active.

Plan/results: `bayesfilter-q20-naf-forward-reverse-{plan,results}-2026-10-06.md`.
Artifacts: `artifacts/q20-naf-forward-reverse-2026-10-06/`.
Terminal audit passes: 229 result artifacts and 5176 source files checked.
The integrated monograph builds to 668 pages; changed rendered pages reviewed.

Shared checkout remains on `preserve/shared-main-before-fab-20260926`, HEAD
`70a6d7e96`. Preserve its unrelated dirty HMC work. Integration and push use
`/tmp/BayesFilter-neutra-q20-20261006`, branch `integrate/neutra-q20-20261006`.
Initial commit `f0feffbfc` is on origin/main. The follow-up contains terminal q20
results, saved-state continuation and final manuscript/accounting updates.
Keep the remote precision APIs, log-domain weight fix and event-monitoring
reference section when syncing; do not blindly copy the shared reference file.

q20 has four inferred parameters, latent dimension 20, T30 UKF target. Use
explicit `tensorflow_eigh_strict_factor_cached`; the constructor's older
custom-op backend stalled in two preserved, charged attempts. Finite-difference
checks passed. Prior importance banks failed the weight screen. Eight saved
SMC populations replayed all 800 target values after chart-Jacobian correction;
these provide an approximate teacher for two known sign regions, not complete
posterior discovery.

Seed 61007: 512 forward/32 reverse updates. Seed 61008: 2048 forward/128 reverse,
then saved-state continuation to 256 reverse updates. All completed maps pass
finite, reload, inverse and declared represented-region checks. The final
1000-point residual median/p99/max is 1.681/57.44/279.99; 76.4% exceed norm one.
This is working training with unresolved whitening, not an admitted q20 map.
Final checkpoint: `continue-reverse-20261006T123655/reverse256-checkpoint.json`.
Teacher: `reuse-smc-20261006T111325`.

The pilot used 6272.79 GPU-process seconds. Local/shared accounting reconciles;
remaining GPU/CPU seconds are 105674.490453/92964.576086. CPU includes 972.54
measured engineering seconds and a separately disclosed 4000-second contingency
for unmetered overhead. Do not charge either again.

Next research action, if continuing the campaign: write the separately priced
matched reverse-rate/budget calibration from the saved forward2048 map, then
confirm with a fresh seed/teacher. Two 512-update reverse arms need about
6687 GPU seconds before setup; the results note proposes an 8400-second
exposure estimate. No long forward ladder or HMC run is queued. Finite residual
errors are repair signals, not a numerical continuation veto or rejection of NAF.
