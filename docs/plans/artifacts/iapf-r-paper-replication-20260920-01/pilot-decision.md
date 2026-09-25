# Pilot decision before fresh-data repeats

2026-09-20. Attempt 01 used the exact paper d5,T100,N0=1000 scope on
data seed 54000005. The complete CPU worker took 8.82 s. iAPF stopped at
zero-based iteration 6, with N=2000; fresh final likelihood ratio to Kalman
was 1.04014. All 600 fits reported numerical convergence. This one draw is
not accuracy or ranking evidence.

The planned objective-degeneracy diagnostic fired: initial fits reached
log variance 10.80 and relative residual .829. Later fits reduced the largest
relative residual to .036 on the final backward pass. The positive-floor
proposal reached .989 original-transition component probability for an
individual particle. These are explanatory findings, not invalid proposals:
the mixture and density-ratio identities remain correct. No parameter hit the
floating-point boundary, and no nonfinite arithmetic occurred. Broadening is
consistent with the equation-(15) degeneracy recorded before execution; the
data do not identify the authors' undocumented optimization workaround.

Decision: continue to the preplanned repeated-likelihood comparison with
the SAME source/controls. Freeze the implementation snapshots in attempt01
before using fresh data seed 55000005. Do not change the fit objective, add
unreviewed scientific bounds, or tune on the replication results. Check d10
capacity on separate pilot data seed 54000010. An inadequate resulting
likelihood distribution rejects this reconstruction's performance claim;
it does not justify claiming the paper has been disproved.

| Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Equation/algorithm identities pass | No continuation veto; large variance/residual are explanatory | Undocumented author floor/solver choices, stochastic likelihood behavior | Fresh-data d5 repeats and separate d10 capacity pilot | Replication of Table 1 or superiority |

Inference status: hard validity checks passed; no statistical ranking;
single likelihood differences descriptive only; no default-readiness;
repeat-level intervals are the next evidence. A full source pass still does
not prove global optimizer quality.
