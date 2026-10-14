# Skeptical review of the refreshed master and executed tranche

Verdict: PASS for the bounded implemented repair phases and their scoped
conclusions. The complete replication program remains incomplete.
This is self-review, not independent review or terminal R6 acceptance.

The review checked wrong baselines, proxy promotion, hidden defaults, unfair
comparisons, stale context, environment differences, missing stop conditions
and commands that could answer the wrong question. Source and finite-fit
contracts are in execution-spec.md; MDPF metrics and strata are in
mdpf-source-and-screen-spec.md; value parity is in value-port-spec.md;
graph diagnosis is in legacy-graph-spec.md.

Material findings were repaired or retained as explicit limits:
- The legacy iAPF consumer computes a different model and time/resampling
  convention. It cannot supply paper-target parity; the new explicit-matrix
  shared kernel was checked at its actual call site.
- The R kappa=1 endpoint violated the mathematical ESS bound through rounding.
  Both reference and candidate now handle that exact endpoint; old fixtures
  remain preserved and v2 fixtures are the authority.
- The saved MDPF tensor was initially aggregated incorrectly. The repaired
  analysis matches the inspected author loss/evaluation aggregation.
- The old graph produces empty outputs. Fresh-process optimizer controls
  localize an interaction; no global optimizer change or eager fallback is
  promoted. Attempt15's later options are not interpreted in isolation.
- Finite-fit stopping, amplitude endpoints, floor choice and reconstruction
  scope are explicit. Stationarity is not a universal filtering veto.
- One checkpoint and a small pilot cannot rank methods. Conditional point
  heuristics do not establish normalized-density performance.
- Twelve dispatcher tests pass, including real supervisor loss and inherited
  timeout/lock behavior, failed-child CLI status, and prevention of false
  scientific completion from value parity.
- Historical seed-field errors in three deterministic diagnostics have an
  additive erratum; original manifests are unchanged.
- The current master replaces stale active instructions with one phase table;
  archived prose cannot reopen closed phases or silently change budgets.

The executable has exact implemented commands, source snapshots, unique
outputs, atomic manifests, external timeouts, inherited locks, recovery,
preflight, dependencies and resource/attempt ceilings. Future unimplemented
phases are visible gaps. validate checks executable structure; it does not
grant scientific completion. status is the current budget authority.

No material uncontrolled default is promoted: original iAPF recipe uncertainty,
failed B_scaled endpoints, missing training-seed inference, missing MDPF density
baselines, FP64-only value scope, and missing fitter/controller/score consumers
all remain explicit. The next phase must preserve these distinctions.

Regression command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python tests/replication/test_repair_master.py
Observed result: 12 tests, PASS, 2.799 seconds.

Validation command:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py validate

Execution command used for ready phases:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py run-ready

GPU launches use escalated device access and verified memory growth.
Routine local repairs use the existing authorization and direct approved
invocations. No new permission rules or broad environment changes were needed.
