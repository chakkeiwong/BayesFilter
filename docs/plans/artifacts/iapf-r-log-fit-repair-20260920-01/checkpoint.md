# Active R iAPF reference checkpoint

2026-09-20. Checkout /home/chakwong/BayesFilter, branch surrogate-hmc,
HEAD 6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. Preserve unrelated dirty files.
Owner requested a careful paper/code audit, reviewed repairs and continuation.
Current bounded audit/repair campaign COMPLETE; no process is running.

Read docs/plans/artifacts/iapf-r-log-fit-repair-20260920-01/result.md first.
Paper/code diagnosis: docs/plans/artifacts/iapf-r-paper-code-audit-20260920-01/result.md.
Plans: docs/plans/iapf-r-paper-code-audit-2026-09-20.md and
       docs/plans/iapf-r-log-fit-repair-2026-09-20.md.

Checked: proposal, weight and backward-recursion identities match the paper.
Strict fitting now rejects nonconvergence/underflow. Saved failures preserve
backward/controller context, and workers execute captured sources.
The d20 relative-loss failures fit roughly 1--2 effective target points. Exact
continuous errors .472332/.261061 exceed their initializers' .175331/.250284;
diagonal target-moment diagnostics give .018480/.017046. This establishes
poor off-cloud fitting in those cases, not failure of every iAPF implementation.

Optional log_quadratic uses full-rank QR and strictly negative curvature,
without clipping or moment fallback. Four pilot and 16 fresh d20,T100 repeats
complete; 12,000 fits pass. Follow-on mean ratio to Kalman .989741, bootstrap95%
[.951537,1.029334]; conditional accuracy screen passes, no heuristic veto.
No statistical method ranking. Equation15 default unchanged. This is an
independent noncanonical R alternative, not full paper replication.
Final tests: seven pytest cases, 1.78s; 88 reference + 33 alternative R checks,
three executed-source mutations and captured-source regressions.

Budget: 1516.719870/1550 repair-worker seconds, all12 launches used;
combined with earlier stage1765.102396/1800 seconds. Follow-on193.390443/200.
Mechanics charged55/120 seconds. Remaining33.280130 worker seconds cannot
support the next stage; no experiment launches remain.

Exact next scientific action under a new bounded allocation: larger untouched
d20 validation of the frozen log fit, then d40/d80 pilots before1000 repetitions
per dimension. Resolve author floor/solver/controller/data ambiguities before
claiming published-result reproduction. TensorFlow debugging follows a
qualified R reference. No nonlinear, score, HMC, KDM or canonical LEDH claim.
