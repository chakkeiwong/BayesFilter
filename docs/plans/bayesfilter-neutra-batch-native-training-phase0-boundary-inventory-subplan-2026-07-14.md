# Phase 0 Subplan: NeuTra Batch-Native Boundary And Inventory

Date: 2026-07-14

Master program:
`docs/plans/bayesfilter-neutra-batch-native-training-knowledge-transfer-master-program-2026-07-14.md`

## Phase Objective

Make the new batch-native policy executable before implementing the fast LGSSM
kernel. No BayesFilter NeuTra optimizer update may proceed unless an actual
repository-owned batch-native target callable is bound to a validated capability.
Inventory all current optimizer entry points and classify nonconforming paths as
migration debt or diagnostic-only.

## Entry Conditions

- The 2026-07-14 `AGENTS.md` and `CLAUDE.md` batch-native policy is active.
- The current LGSSM exact adapter is known to use `tf.map_fn` over scalar rows.
- Historical artifacts are preserved but provide no batch-native admission.
- No Phase 7 target-specific serious training is active.

## Required Artifacts

- `bayesfilter/inference/neutra_batching.py`: repository-owned capability type,
  callable binding/factory, validation, and runtime audit metadata.
- Focused changes to `bayesfilter/inference/neutra_training.py` that reject
  ineligible routes before output-directory creation.
- Focused tests for valid batch-native and invalid missing/scalar/row-mapped/
  batch-size-one/self-attested routes.
- Inventory artifact under
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase0/`.
- Phase 0 result and reviewed Phase 1 subplan under `docs/plans`.

## Capability Contract

The repository-owned capability must bind:

- target and adapter identity;
- backend identifier and actual bound callable;
- minimum batch size greater than one;
- `scalar_fallback_used=False`;
- `sample_axis_python_loop_used=False`;
- `row_mapped_scalar_target_used=False`;
- XLA readiness;
- value, score, and optional status output roles; and
- source/evidence path.

The trainer must call this bound batch method directly. A caller-provided string
or dictionary is not authority. Generic test adapters may obtain capability only
through the repository factory and must expose a genuinely vectorized batch
method.

## Required Checks

1. Python compile checks for changed modules/tests.
2. Unit tests proving eligible vectorized Gaussian training still works.
3. Negative tests for batch size one, missing capability, scalar fallback,
   sample loop, row-mapped scalar target, invalid XLA flag, and detached callable.
4. Test that rejection occurs before output-directory creation or adapter call.
5. Static inventory of optimizer-update entry points and classification review.
6. Source check that the capability implementation contains no NumPy or host
   callback.
7. Existing focused NeuTra training tests after fixture migration.
8. `git diff --check` for touched files.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Does repository enforcement prevent non-batch-native targets from reaching a NeuTra optimizer update? |
| Pass criterion | Every negative adapter is rejected before output creation/call, and a vectorized batch adapter passes existing deterministic training checks. |
| Hard veto | An ineligible target reaches an update, a caller can self-attest, output is created before rejection, or existing eligible training semantics change. |
| Explanatory only | Inventory counts and historical route descriptions. |
| Artifact | Focused test output, inventory JSON/Markdown, and Phase 0 result note. |
| Nonclaims | Phase 0 does not make the LGSSM target batch-native, faster, scientifically valid, or ready for serious training. |

## Forbidden Claims And Actions

- Do not describe the current exact LGSSM adapter as eligible.
- Do not run any optimizer update through its row-mapped scalar batch method.
- Do not change target, likelihood, score, status, seed, or optimizer math.
- Do not delete or rewrite historical artifacts.
- Do not treat metadata labels as proof of callable behavior.
- Do not launch GPU training in Phase 0.

## Default And Assumption Audit

| Choice | Provenance | Risk | Diagnostic | Status |
| --- | --- | --- | --- | --- |
| Fail before output directory | No-overwrite/evidence integrity policy | Some callers may expect config artifacts on failure | Negative test checks no path exists | reviewed boundary |
| Repository factory issues capability | Canonical identity patterns | Factory could still accept a misleading callable | Bind method identity, require explicit vectorized method, audit source/inventory | implementation hypothesis |
| Existing scalar HMC APIs remain | HMC needs scalar target and policy addresses training only | Accidental trainer fallback | Trainer accepts only bound batch callable | frozen compatibility boundary |
| Test Gaussian adapter is vectorized | Existing test equations already use batch reductions | Tests could pass without exercising contract | Explicit capability and call-count assertions | reviewed fixture choice |

## Skeptical Subplan Audit

- Wrong baseline: no numerical candidate is compared in Phase 0; current policy
  and trainer entry points are the baseline.
- Proxy promotion: metadata alone is explicitly insufficient; actual binding and
  negative execution tests are required.
- Missing stop: stop on enforcement bypass, output-before-rejection, callable
  detachment, or compatibility failure that cannot be locally repaired.
- Hidden scope: inventory includes testing and benchmark optimizer paths, not
  just the active strict harness.
- Artifact fitness: inventory records path, function, update mechanism, batch
  behavior, classification, and required migration.

Audit verdict: **PASS**. Phase 0 is bounded, testable, and required before any
new numerical work can accidentally violate policy.

## Exact Next-Phase Handoff Conditions

Phase 1 may start when:

- the generic trainer fails closed on every ineligible route;
- eligible vectorized test training passes;
- the inventory is complete enough to name every discovered optimizer path;
- Phase 0 result records unresolved migration debt without claiming it fixed;
- the Phase 1 transfer subplan cites exact DSGE and BayesFilter source anchors;
  and
- the Phase 1 subplan suitability review finds no real blocker.

## Stop Conditions

Stop only if enforcement cannot be added without changing public scalar HMC
semantics, an unrelated dirty change overlaps the exact edited lines and cannot
be preserved, the inventory reveals an active serious training process, or the
focused compatibility suite exposes a material public-API decision requiring
owner direction. Otherwise repair and continue.

## Phase-End Procedure

1. Run all required local checks.
2. Write the Phase 0 result/close record.
3. Draft or refresh the Phase 1 knowledge-transfer subplan.
4. Review Phase 1 for consistency, correctness, feasibility, artifact coverage,
   default/assumption coverage, and boundary safety.
5. Continue if no real blocker exists.

