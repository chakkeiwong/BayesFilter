> Superseded for active replication work on 29 September 2026 by
> [controlled-SMC/KDM checkpoint](controlled-smc-kdm-checkpoint.md).
> The owner authorized execution. The following is preserved iAPF history;
> its old next action is deferred, not active authority.

# iAPF / MDPF replication active checkpoint

Updated 28 September 2026 Hong Kong. Repo /home/chakwong/BayesFilter;
branch surrogate-hmc, base 89065bc6354801cb368d5e163ba49fd9c4372d10.
Preserve unrelated dirty work. Both scientific replications INCOMPLETE.
Master: docs/plans/iapf-kdm-replication-master-program-2026-09-24.md.
Plan: docs/plans/iapf-mdpf-whole-repair-plan-20260928.md.
Evidence root: docs/plans/artifacts/iapf-mdpf-whole-repair-20260928-01.
Current: r4d-result.md, r4d-review.md, status-after-r4d.json.
R4c and attempts 01–28 remain historical evidence for their own source snapshots.

R4d DONE as bounded repair/diagnostics; paper-horizon numerical parity FAIL.
- Shared _vmmin loop backs off nonfinite trial values and mirrors R lower Hessian.
- 37 semantic controls pass on CPU, GPU graph, GPU XLA snapshots.
- Attempt 33 caught added diagnostic keys missing from backward-loop empty state;
  repaired, also fixed no-change negative rejected-trial counter.
- Attempt 34: 43 semantic/diagnostic XLA checks PASS.
- Attempts 35/36: 129 actual small fitter/backward/controller checks each on
  CPU/GPU XLA PASS on current numerical source.
- Attempt 37: 92/95 paper checks pass; backward pass 2 centers/cov/logfloors fail
  .00175473/.00176944/.00072073, unchanged 1e-5 absolute+relative criterion.
  Five XLA signatures trace once. Full adaptive run withheld by strict harness.
- Exact R initializer still leaves isolated time 54 parameter error
  CPU .000184638, GPU graph .00295567, GPU XLA .00314469; divergence exceeds 1e-8
  around accepted iteration 83/84. All finite, 100-step limit.
- R4c attempt 28 full actual d5,T100 adaptive consumer completed on its older
  source snapshot: 8 calls, 6 fits, stop 6, N=2000, logZ delta R+.002201253633.
  It does not certify repaired full-horizon behavior.
- R4c same-backend log-target perturbation~1e-14 moved coefficients~.003.
  CSV precision, initializer alone and XLA alone do not explain all divergence.
- Original article recipe remains unrecovered; finite unpenalized recipe is
  a labeled reconstruction. Paper Proposition 1/Section 5.1 checked again.

Next exact research action: define R4d-next paired downstream evidence contract
before numerical execution: calibration-only numerical error margin against
Monte Carlo error; freeze protocol; untouched paired data/streams, exact Kalman,
BPF/fully adapted comparators, uncertainty and full adaptation cost. Preserve old
parameter FAIL. No validated downstream margin or numerical remedy exists yet.
Follow-up source audit: docs/reference/iapf-followup-methods-20260928.md (+ JSON).
Controlled SMC supplies an explicit log-quadratic fit and author MATLAB code;
annealed/forward-only methods motivate coverage and backward-error diagnostics.
Consider a separately named controlled-SMC comparison in R4d-next; preserve the
original iAPF target. Latest annealed revision and numerical code parity remain
unchecked. Literature review used no GPU/numerical campaign budget.
Proposed successor: docs/plans/controlled-smc-kdm-replication-master-program-20260928.md.
Owner asked whether to prioritize controlled SMC instead of iAPF. Draft reviewed
as a migration proposal; not activated, not executable, no numerical launch.
It preserves iAPF incomplete status and independent KDM work; active budget unchanged.
Investigate a concrete new source mismatch if identified; no blind rerun.
R3b normalized MDPF density comparators remains independent eligible implementation.
R3c Appendix B has unrecovered coefficients/initialization/schedule; label hypotheses.
Full MDPF 10 additional trials~85.89h at observed 8.589h each, under-budgeted.
Score/HMC/LEDH are later extensions, not filtering-replication prerequisites.

37/38 attempts accounted: 26 COMPLETE, 11 FAILED; no live job/no ready numerical phase.
CPU charged 1046.833064/1800sec, remaining 753.166936.
GPU charged 42748.699649/173700sec, remaining 36.375361h.
R4d used 29.178210 CPU sec and 218.004948 GPU sec; within 240/1800 sec ceilings.
A new bounded tranche needs actual reviewed contract/code, no renewed permission
under unchanged parent scope/budget. Fifteen dispatcher tests PASS (2.810sec).
No independent agent, external message, package mutation or default promotion.
