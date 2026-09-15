# RECOVERY MEMO PART 2 — decision, files, next actions

Continues `ledh-surrogate-hmc-RECOVERY-2026-09-13.md`. Read that first.

---

## The decision that blocks progress

The question changed. It is no longer *"how do we afford a 14.6-day run"* but
*"is there a run worth affording"*. Four options:

| # | option | cost | what it buys |
|---|---|---|---|
| **A** | re-parameterize damping onto iteration-count knobs (`correction_steps`, `pairwise_steps`), then sweep | knob table already measured; re-planned sweep | a sweep with a real independent variable **and** a real 32–37% compute saving to trade against mixing loss |
| B | sweep the magnitude knobs anyway | 14.6 days | a vacuous pass — arms differ by ~3e-05 in force |
| C | drop the damping sweep from Phase 4a | nothing | Phase 4a reverts to plain LGSSM certification; the *cheap-force* half of Corollary 5.2 goes untested |
| D | test Corollary 5.2 **invariance only**, not its benefit | short run | confirms the θ-marginal is force-invariant using a deliberately crude force (e.g. `correction_steps=0`, which does change the computation) without claiming any speedup |

**A is the only option that tests what the campaign set out to test.** Its
prerequisite is already done: the knob table shows `correction_steps` 4→0 saves
37% and `pairwise_steps` 4→0 saves 32%, both far outside the ±7% noise floor.

**D is cheap and worth doing regardless.** The invariance half of Corollary 5.2
is a claim about the Metropolis correction and is testable on a short chain. It
would validate the adapter end-to-end without pretending the magnitude knobs
deliver a speedup.

This is an owner decision (scope / materially expanded compute), not an
implementation choice. Nothing was silently re-parameterized.

---

## What is NOT concluded

- Corollary 5.2 is **not** disproved. It is a statement about the Metropolis
  correction; nothing measured here bears on its mathematics.
- The dual-parameter adapter is **not** wrong. It computes exact value +
  analytical biased score, verified bitwise and against the analytical reference
  at two scales.
- Surrogate-force HMC is **not** dead as a direction. It needs a damping knob
  that reduces work — and two such knobs now have measured savings.
- No damping ranking, W₁/W₂ agreement, posterior correctness, or HMC convergence
  claim is supported. **No HMC chain was ever run to completion in this session.**
- The knob table does **not** show any knob is *safe*. A cheaper force may
  collapse acceptance. That is what a re-parameterized sweep would measure.
- Per the LEDH Per-Scope Tuning Rule, no per-scope tuning artifact exists for
  (LGSSM d=3, T=50, N=252, contract_e, float64, dual-parameter target), so this
  scope carries **no per-model claim**.

---

## Files created this session

| path | purpose |
|---|---|
| `docs/benchmarks/ledh_cost_structure_probe.py` | cost law (N / substeps / T) + eager-vs-graph |
| `docs/benchmarks/ledh_graph_mode_gradient_verification.py` | V1/V2/V3 surrogate-gradient contract under `tf.function` |
| `docs/benchmarks/ledh_value_direction_invariance_parity.py` | K=P vs K=1 value parity (bitwise) |
| `docs/benchmarks/ledh_cheap_force_knob_search.py` | which knob is actually cheaper (**ignore its "viable: 0" verdict; gate too strict**) |
| `docs/benchmarks/ledh_plan_scale_graph_probe.py` | plan-scale eager/graph (**has the AutoGraph free-variable bug; prefer the verification script**) |
| `docs/benchmarks/ledh_dual_target_scaling_probe.py` | first N-scaling probe (superseded by the cost-structure probe) |
| `docs/plans/ledh-surrogate-hmc-phase4a-execution-status-2026-09-13.md` | full defect detail 1–10 |
| `docs/plans/ledh-surrogate-hmc-RECOVERY-2026-09-13.md` + this part 2 | recovery entry point |

Raw results: `/tmp/ledh_cost_structure.json`, `/tmp/ledh_knob_search.json`,
`/tmp/ledh_graph_gradient_verify.json`, `/tmp/ledh_verify_T50.json`,
`/tmp/ledh_damping_premise.json`. Logs: `/tmp/{cost_structure,knob_search,
graph_grad_verify,verify_T50,premise_check,T50_graph}.log`.
**`/tmp` will not survive a reboot — copy anything needed into
`docs/plans/artifacts/` before then.**

## Files modified

- `bayesfilter/inference/ledh_dual_parameter_target.py` — K=1 value path
  (`__call__` and `value_only`); new `as_graph_callable(param_dim)`;
  `_graph_callable` cache field.
- `docs/benchmarks/ledh_surrogate_hmc_damping_calibration_pilot.py` — memory
  growth fail-closed; `EXECUTION_MODE` (default `graph`); flattened `np.savez`;
  run manifest; `marginal_w1`; `score_support` + live/dead coordinate handling;
  `SMOKE_*` / `OUTPUT_DIR` overrides; corrected `sample_chain` comment.

Nothing committed. Working tree is dirty by design; `AGENTS.md` and `CLAUDE.md`
were already modified before this session — **not mine, leave them.**

---

## Reproduce the key measurements

```bash
# All runs: graph mode is the default. 4080 SUPER is GPU index 1 under PCI_BUS_ID.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

# cost law + eager/graph  (~15 min)
conda run -n tftwogpu --no-capture-output python -u \
  docs/benchmarks/ledh_cost_structure_probe.py

# gradient contract at plan horizon  (~15 min; T=50 trace is ~8 min)
VERIFY_N=252 VERIFY_T=50 VERIFY_SUBSTEPS=8 VERIFY_SINKHORN=8 \
conda run -n tftwogpu --no-capture-output python -u \
  docs/benchmarks/ledh_graph_mode_gradient_verification.py

# knob search  (~10 min)
conda run -n tftwogpu --no-capture-output python -u \
  docs/benchmarks/ledh_cheap_force_knob_search.py
```

**Operational notes learned the hard way:**

- Always `python -u` and redirect to a file. `grep | tail` buffers the entire
  stream until exit, so a multi-hour run shows *zero* output (defect 10).
- Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Without it, `CUDA_VISIBLE_DEVICES=1`
  selects the 5080, not the 4080 SUPER (defect 2).
- N=1008 OOMs in eager mode; graph mode was never tested at that scale. Use N=252
  for T=50 pending a proper graph-mode capacity check.
- `nvidia-smi --query-compute-apps` returns empty under WSL2 — it cannot
  enumerate processes inside the VM. Infer ownership by killing and re-reading
  `memory.used`.

---

## Next actions, in order

1. **Run the value-vs-score discriminating test** (part 1, "Two findings that are
   NOT about damping", item 2). Compare the *value* at `correction_steps=4` vs
   `0`. Minutes to run; the only open question that could indicate a defect in
   the value/score seam rather than a property of the fixture. Designed but not
   run.
2. **Owner decision on A / B / C / D above.** Blocking; everything downstream
   depends on it.
3. **Re-tune `reset_sinkhorn_steps`** independently of damping. 8→2 changes the
   score by 4.4e-09, so ~6 of 8 iterations are waste in the *exact* baseline
   (~9% of score cost). Confirm at larger N before touching a default.
4. **Repair defect 9** so the adapter has real coverage of the Corollary 5.2
   properties.
5. **Decide the θ dimension** (defect 7). Either the fixture should identify
   θ[3:], or the sweep should sample a 3-vector — which also cuts the score from
   K=5 to K=3 passes, a further ~33% saving. Scope question, not an
   implementation detail.
6. Only then launch a sweep, at whatever scale and parameterization is
   authorized, and record results with an inference-status table.

---

## Session end state

- **No HMC chain completed. No sweep launched. No scientific claim made.**
- Graph mode implemented, verified at two scales, and default.
- Working tree dirty by design. Nothing committed (per owner standing instruction).
  launched. Background jobs from this session are gone — nothing needs killing,
  but confirm with `pgrep -af ledh_` in a fresh session.
