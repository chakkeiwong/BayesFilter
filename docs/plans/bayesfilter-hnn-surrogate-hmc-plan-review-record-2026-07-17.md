# Corrected Neural-Force HMC Program Review Record

Date: 2026-07-17

Reviewed path:
`docs/plans/bayesfilter-hnn-surrogate-hmc-master-program-2026-07-17.md`.

Reviewer role: Claude Opus, max effort, bounded read-only advisory review.
Codex remained supervisor and final authority.

## Local Skeptical Audit

The local audit explicitly checked wrong baselines, proxy promotion, missing
stop conditions, unfair comparisons, hidden defaults, stale NeuTra evidence,
environment mismatch, artifact sufficiency, filtering-target identity, the
NeuTra chart Jacobian, budget arithmetic, repair semantics, and model counts.

Material repairs before Claude review:

1. Corrected the Tier B count and then replaced prose with an exact eight-row
   configuration/budget matrix.
2. Restored tuned raw-coordinate plain HMC as a mandatory baseline, allowing
   preserved matching evidence only after identity and diagnostic replay.
3. Bound endpoint correction in NeuTra coordinates to
   `U_F(T(z))-log|det dT/dz|`; the raw target alone is explicitly invalid.
4. Reconciled the CPU phase allocation with the 22-hour program ceiling.

## Claude Round 1

Verdict: `VERDICT: REVISE`.

Material findings:

1. The computational-viability question lacked a binding decision rule.
2. Tier A prose overstated uniformity despite qualified/noncentral historical
   evidence in several rows.
3. Tier B scope-to-budget mapping was ambiguous.
4. A valid mixture kernel could blur failure of the pure HNN candidate.
5. The prospective sampler thresholds admitted an unconstrained owner
   adjudication.

Repairs:

- validity and performance are now orthogonal classifications;
- a predeclared descriptive performance screen uses seconds per minimum bulk
  ESS in reuse-scenario and sampling-only ledgers, without a superiority claim;
- Tier A states its heterogeneous historical evidence and applies one fresh
  prospective contract;
- all eight Tier B configurations have explicit 6-GPU-hour ceilings including
  their one repair attempt;
- mixture results can receive `MIXTURE_VALIDATED` but never
  `HNN_VALIDITY_CONFIRMED`;
- R-hat/ESS gates are hard, and a later threshold change receives only
  `QUALIFIED_POSTHOC_ADJUDICATION`.

## Claude Round 2

Verdict: `VERDICT: AGREE`.

Claude confirmed that all five findings were resolved and found no newly
introduced material scientific, mathematical, feasibility, budget, evidence,
or phase-handoff defect.

## Final Decision

`REVIEWED_READY_FOR_P0`.

This decision admits the program for P0 scope/default/command freezing. It does
not admit the kernel implementation, training harness, target cells, serious
GPU runs, validity, or performance. Those remain phase evidence gates.
